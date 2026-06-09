/*****************************************************************************\
 *  gres_hami.c - Support HAMi-core as a generic resource for GPU sharing.
 *
 *  HAMi-core provides true GPU memory isolation and SM compute limiting via
 *  LD_PRELOAD interception of CUDA API calls. This plugin exposes GPU memory
 *  as a shareable GRES resource so that slurmctld can:
 *    - Track remaining virtual memory per GPU (bin-packing via cons_tres)
 *    - Inject HAMi-core environment variables into job steps automatically
 *
 *  gres.conf example (Count = total GPU memory, supports K/M/G/T suffixes):
 *    Name=hami Count=40G File=/dev/nvidia0   # A100 40 GB
 *    Name=hami Count=80G File=/dev/nvidia1   # H100 80 GB
 *
 *  Submission example (request 8 GB of GPU memory):
 *    sbatch --gres=hami:8G train.sh
 *
 *  Environment variables injected per job:
 *    LD_PRELOAD                     - prepends libvgpu.so
 *    CUDA_DEVICE_MEMORY_LIMIT_{i}   - hard memory cap in bytes per visible GPU
 *    CUDA_DEVICE_MEMORY_SHARED_CACHE - per-job shared-region file path
 *
 *  SM utilisation limit (CUDA_DEVICE_SM_LIMIT) is NOT set by this plugin so
 *  that users may control it themselves via their job script or --export.
 *  HAMi-core defaults to 100 % when the variable is absent.
 *****************************************************************************
 *  Based on gres_mps.c and gres_shard.c
 *  Copyright (C) SchedMD LLC.
 *
 *  This file is part of Slurm, a resource management program.
 *  For details, see <https://slurm.schedmd.com/>.
 *  Please also read the included file: DISCLAIMER.
 *
 *  Slurm is free software; you can redistribute it and/or modify it under
 *  the terms of the GNU General Public License as published by the Free
 *  Software Foundation; either version 2 of the License, or (at your option)
 *  any later version.
 *
 *  In addition, as a special exception, the copyright holders give permission
 *  to link the code of portions of this program with the OpenSSL library under
 *  certain conditions as described in each individual source file, and
 *  distribute linked combinations including the two. You must obey the GNU
 *  General Public License in all respects for all of the code used other than
 *  OpenSSL. If you modify file(s) with this exception, you may extend this
 *  exception to your version of the file(s), but you are not obligated to do
 *  so. If you do not wish to do so, delete this exception statement from your
 *  version.  If you delete this exception statement from all source files in
 *  the program, then also delete it here.
 *
 *  Slurm is distributed in the hope that it will be useful, but WITHOUT ANY
 *  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 *  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 *  details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with Slurm; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA.
\*****************************************************************************/

#define _GNU_SOURCE

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "slurm/slurm.h"
#include "slurm/slurm_errno.h"

#include "src/common/slurm_xlator.h"
#include "src/common/bitstring.h"
#include "src/common/env.h"
#include "src/interfaces/gres.h"
#include "src/common/hostlist.h"
#include "src/common/list.h"
#include "src/common/xmalloc.h"
#include "src/common/xstring.h"

#include "../common/gres_common.h"
#include "../common/gres_c_s.h"

/* Required Slurm plugin symbols */
const char plugin_name[]    = "Gres HAMi plugin";
const char plugin_type[]    = "gres/hami";
const uint32_t plugin_version = SLURM_VERSION_NUMBER;

/*
 * Path to the HAMi-core interception library. Override at build time with
 * -DHAMI_LIB_PATH=\"/your/path/libvgpu.so\" if needed.
 */
#ifndef HAMI_LIB_PATH
#define HAMI_LIB_PATH "/usr/local/lib/hami/libvgpu.so"
#endif

/*
 * Directory used for per-job HAMi-core shared-region cache files.
 * Each job gets its own file: <HAMI_CACHE_DIR>/hami-job<jobid>-gpu<id>.cache
 */
#ifndef HAMI_CACHE_DIR
#define HAMI_CACHE_DIR "/tmp"
#endif

static list_t *gres_devices = NULL;
static uint32_t node_flags  = 0;

/* ── plugin lifecycle ──────────────────────────────────────────────────── */

extern int init(void)
{
	debug("loaded");
	return SLURM_SUCCESS;
}

extern void fini(void)
{
	debug("unloading");
	FREE_NULL_LIST(gres_devices);
	gres_c_s_fini();
}

/* ── node config ───────────────────────────────────────────────────────── */

/*
 * Load and validate gres.conf entries for gres/hami.
 * hami is a "shared" GRES that rides on top of gres/gpu (the "sharing" GRES),
 * exactly the same relationship that gres/mps and gres/shard have with gpu.
 */
extern int gres_p_node_config_load(list_t *gres_conf_list,
				   node_config_load_t *config)
{
	int rc = gres_c_s_init_share_devices(
		gres_conf_list, &gres_devices, config, "gpu");

	if (rc != SLURM_SUCCESS)
		return rc;

	node_flags = 0;
	(void) list_for_each(gres_conf_list,
			     gres_common_set_env_types_on_node_flags,
			     &node_flags);
	return rc;
}

/* ── environment injection ─────────────────────────────────────────────── */

/*
 * Return the total bytes configured for the device identified by global_id.
 * Used for informational / validation purposes.
 */
static uint64_t _get_dev_bytes(int global_id)
{
	list_itr_t *itr;
	shared_dev_info_t *info;
	uint64_t count = NO_VAL64;

	if (!shared_info) {
		error("%s: shared_info is NULL", __func__);
		return 0;
	}

	itr = list_iterator_create(shared_info);
	while ((info = list_next(itr))) {
		if (info->id == global_id) {
			count = info->count;
			break;
		}
	}
	list_iterator_destroy(itr);

	if (count == NO_VAL64) {
		error("%s: no hami device found for global_id %d",
		      __func__, global_id);
		return 0;
	}
	return count;
}

/*
 * Prepend libvgpu.so to LD_PRELOAD, preserving any value already present.
 */
static void _set_ld_preload(char ***env_ptr)
{
	char *existing = getenvp(*env_ptr, "LD_PRELOAD");
	char *new_val;

	if (existing && existing[0] != '\0')
		new_val = xstrdup_printf("%s:%s", HAMI_LIB_PATH, existing);
	else
		new_val = xstrdup(HAMI_LIB_PATH);

	env_array_overwrite(env_ptr, "LD_PRELOAD", new_val);
	xfree(new_val);
}

/*
 * Core environment setup called from job_set_env, step_set_env, task_set_env.
 *
 * Sets:
 *   CUDA_VISIBLE_DEVICES            (via gres_common_gpu_set_env)
 *   LD_PRELOAD                      (prepend libvgpu.so)
 *   CUDA_DEVICE_MEMORY_LIMIT_{i}    (gres_cnt/num_gpus bytes, per visible GPU)
 *   CUDA_DEVICE_MEMORY_SHARED_CACHE (per-job file for HAMi-core multiprocess
 *                                    coordination within a single job;
 *                                    cross-job capacity is managed by Slurm)
 *
 * CUDA_DEVICE_SM_LIMIT is injected only if the user has already placed it in
 * the job environment (via --export or the gpu-submit wrapper).  If absent,
 * HAMi-core defaults to 100 % SM (no throttling).
 */
static void _set_env(common_gres_env_t *gres_env)
{
	char buf[256];
	uint64_t dev_total_bytes;

	/* global_id starts unknown; gres_common_gpu_set_env fills it in */
	gres_env->global_id    = -1;
	gres_env->gres_conf_flags = node_flags | GRES_CONF_ENV_NVML;
	gres_env->gres_devices = gres_devices;
	gres_env->prefix       = "";

	/* Sets CUDA_VISIBLE_DEVICES / NVIDIA_VISIBLE_DEVICES and global_id */
	gres_common_gpu_set_env(gres_env);

	if (!gres_env->gres_cnt) {
		/*
		 * gres_cnt == 0 means the GRES is being unset (e.g. the job
		 * requested --gres=none within an allocation).  Clear our vars.
		 */
		unsetenvp(*gres_env->env_ptr, "LD_PRELOAD");
		unsetenvp(*gres_env->env_ptr, "CUDA_DEVICE_MEMORY_SHARED_CACHE");
		/* clear per-GPU limit vars (CUDA_DEVICE_MEMORY_LIMIT_0 ... _N) */
		{
			int num_gpus = gres_env->bit_alloc ?
				       bit_set_count(gres_env->bit_alloc) : 1;
			if (num_gpus < 1)
				num_gpus = 1;
			for (int i = 0; i < num_gpus; i++) {
				char var_name[48];
				snprintf(var_name, sizeof(var_name),
					 "CUDA_DEVICE_MEMORY_LIMIT_%d", i);
				unsetenvp(*gres_env->env_ptr, var_name);
			}
		}
		return;
	}

	/* Sanity-check: allocated bytes must not exceed the device total */
	if (gres_env->global_id >= 0) {
		dev_total_bytes = _get_dev_bytes(gres_env->global_id);
		if (dev_total_bytes > 0 && gres_env->gres_cnt > dev_total_bytes) {
			error("%s: job requested %"PRIu64" bytes but device %d only"
			      " has %"PRIu64" bytes total; capping",
			      __func__, gres_env->gres_cnt,
			      gres_env->global_id, dev_total_bytes);
			gres_env->gres_cnt = dev_total_bytes;
		}
	}

	/* LD_PRELOAD: inject HAMi-core */
	_set_ld_preload(gres_env->env_ptr);

	/*
	 * CUDA_DEVICE_MEMORY_LIMIT_{i}: per-visible-device memory limit.
	 *
	 * HAMi-core applies CUDA_DEVICE_MEMORY_LIMIT_0/_1/... per visible GPU
	 * (indexed by position in CUDA_VISIBLE_DEVICES).  Setting only the
	 * generic CUDA_DEVICE_MEMORY_LIMIT would apply the full gres_cnt to
	 * every visible GPU, giving a multi-GPU job N× the intended quota.
	 *
	 * We split gres_cnt equally across all allocated devices (bits set in
	 * bit_alloc).  Slurm's cons_tres bin-packing guarantees each device
	 * contributes at most its configured Count, so equal split is correct
	 * when all allocated devices have the same Count.
	 */
	{
		int num_gpus = gres_env->bit_alloc ?
			       bit_set_count(gres_env->bit_alloc) : 1;
		if (num_gpus < 1)
			num_gpus = 1;

		uint64_t per_gpu_bytes =
			(gres_env->gres_cnt / (uint64_t)num_gpus);

		for (int i = 0; i < num_gpus; i++) {
			char var_name[48];
			snprintf(var_name, sizeof(var_name),
				 "CUDA_DEVICE_MEMORY_LIMIT_%d", i);
			snprintf(buf, sizeof(buf), "%"PRIu64, per_gpu_bytes);
			env_array_overwrite(gres_env->env_ptr, var_name, buf);
		}
	}

	/*
	 * CUDA_DEVICE_MEMORY_SHARED_CACHE: per-job coordination file.
	 *
	 * Each job gets its own cache file so that HAMi-core's limit is
	 * initialised independently per job.  Cross-job capacity is managed
	 * by Slurm (cons_tres bin-packing); HAMi-core only needs to enforce
	 * the per-job limit within the job's own processes.
	 *
	 * File name: hami-job<jobid>-gpu<id>.cache
	 * Cleaned up by the Epilog script (hami_epilog.sh) after job completion.
	 */
	{
		const char *job_id_str =
			getenvp(*gres_env->env_ptr, "SLURM_JOB_ID");
		uint32_t job_id;

		if (job_id_str) {
			job_id = (uint32_t)atol(job_id_str);
		} else {
			error("%s: SLURM_JOB_ID not set; cache isolation disabled",
			      __func__);
			job_id = 0;
		}

		if (gres_env->global_id >= 0)
			snprintf(buf, sizeof(buf), "%s/hami-job%u-gpu%d.cache",
				 HAMI_CACHE_DIR, job_id, gres_env->global_id);
		else
			snprintf(buf, sizeof(buf), "%s/hami-job%u-gpu-unknown.cache",
				 HAMI_CACHE_DIR, job_id);
	}

	env_array_overwrite(gres_env->env_ptr,
			    "CUDA_DEVICE_MEMORY_SHARED_CACHE", buf);

	/*
	 * CUDA_DEVICE_SM_LIMIT: honour whatever the user injected via
	 * --export=...,CUDA_DEVICE_SM_LIMIT=50 or the gpu-submit wrapper.
	 * If absent, leave unset so HAMi-core defaults to 100 % (no throttle).
	 * We also enable the utilisation watcher when a limit is present.
	 */
	if (getenvp(*gres_env->env_ptr, "CUDA_DEVICE_SM_LIMIT")) {
		/* Ensure HAMi-core's utilization watcher thread activates */
		if (!getenvp(*gres_env->env_ptr, "GPU_CORE_UTILIZATION_POLICY"))
			env_array_overwrite(gres_env->env_ptr,
					    "GPU_CORE_UTILIZATION_POLICY",
					    "FORCE");
	}
}

/* ── gres plugin entry points ──────────────────────────────────────────── */

/*
 * Set environment variables for the job (all tasks on a node).
 */
extern void gres_p_job_set_env(char ***job_env_ptr,
			       bitstr_t *gres_bit_alloc,
			       uint64_t gres_per_node,
			       gres_internal_flags_t flags)
{
	common_gres_env_t gres_env = {
		.bit_alloc  = gres_bit_alloc,
		.env_ptr    = job_env_ptr,
		.flags      = flags,
		.gres_cnt   = gres_per_node,
		.is_job     = true,
	};
	_set_env(&gres_env);
}

/*
 * Set environment variables for a job step (srun).
 */
extern void gres_p_step_set_env(char ***step_env_ptr,
				bitstr_t *gres_bit_alloc,
				uint64_t gres_per_node,
				gres_internal_flags_t flags)
{
	common_gres_env_t gres_env = {
		.bit_alloc  = gres_bit_alloc,
		.env_ptr    = step_env_ptr,
		.flags      = flags,
		.gres_cnt   = gres_per_node,
	};
	_set_env(&gres_env);
}

/*
 * Set environment variables per task (when --ntasks-per-gpu or similar
 * task-level GRES binding is in effect).
 */
extern void gres_p_task_set_env(char ***task_env_ptr,
				bitstr_t *gres_bit_alloc,
				uint64_t gres_cnt,
				bitstr_t *usable_gres,
				gres_internal_flags_t flags)
{
	common_gres_env_t gres_env = {
		.bit_alloc   = gres_bit_alloc,
		.env_ptr     = task_env_ptr,
		.flags       = flags,
		.gres_cnt    = gres_cnt,
		.is_task     = true,
		.usable_gres = usable_gres,
	};
	_set_env(&gres_env);
}

/* ── stepd serialisation ───────────────────────────────────────────────── */

/* Send GRES information from slurmd to slurmstepd */
extern void gres_p_send_stepd(buf_t *buffer)
{
	gres_send_stepd(buffer, gres_devices);

	pack32(node_flags, buffer);
	
	gres_c_s_send_stepd(buffer);
	return;
}

/* Receive GRES information in slurmstepd */
extern void gres_p_recv_stepd(buf_t *buffer)
{
	gres_recv_stepd(buffer, &gres_devices);

	safe_unpack32(&node_flags, buffer);

	gres_c_s_recv_stepd(buffer);
	return;

unpack_error:
	error("%s: failed to unpack stepd data", __func__);
}

/* ── misc plugin entry points ──────────────────────────────────────────── */

extern list_t *gres_p_get_devices(void)
{
	return gres_devices;
}

extern void gres_p_step_hardware_init(bitstr_t *usable_gres, char *settings)
{
	return;
}

extern void gres_p_step_hardware_fini(void)
{
	return;
}

/* ── prolog / epilog environment ───────────────────────────────────────── */

/*
 * Build the record used to set environment variables in job prolog/epilog.
 * Mirrors gres_mps.c / gres_shard.c exactly.
 */
extern gres_prep_t *gres_p_prep_build_env(gres_job_state_t *gres_js)
{
	int i;
	gres_prep_t *gres_prep;

	gres_prep = xmalloc(sizeof(gres_prep_t));
	gres_prep->node_cnt = gres_js->node_cnt;
	gres_prep->gres_bit_alloc = xcalloc(gres_prep->node_cnt,
					    sizeof(bitstr_t *));
	gres_prep->gres_cnt_node_alloc = xcalloc(gres_prep->node_cnt,
						 sizeof(uint64_t));
	for (i = 0; i < gres_prep->node_cnt; i++) {
		if (gres_js->gres_bit_alloc &&
		    gres_js->gres_bit_alloc[i]) {
			gres_prep->gres_bit_alloc[i] =
				bit_copy(gres_js->gres_bit_alloc[i]);
			gres_prep->gres_cnt_node_alloc[i] =
				gres_js->gres_cnt_node_alloc[i];
		}
	}
	return gres_prep;
}

/*
 * Set environment variables for job prolog/epilog scripts.
 * Exposes the same HAMi-core variables so that prolog scripts can, e.g.,
 * pre-create the shared-region cache file with correct permissions.
 */
extern void gres_p_prep_set_env(char ***prep_env_ptr,
				gres_prep_t *gres_prep, int node_inx)
{
	int dev_inx = -1, global_id = -1, i;
	gres_device_t *gres_device;
	list_itr_t *iter;
	char buf[256];

	/* Set CUDA_VISIBLE_DEVICES for prolog context */
	if (gres_common_prep_set_env(prep_env_ptr, gres_prep, node_inx,
				     node_flags | GRES_CONF_ENV_NVML,
				     gres_devices))
		return;

	/* Resolve which physical GPU was allocated */
	if (gres_prep->gres_bit_alloc &&
	    gres_prep->gres_bit_alloc[node_inx])
		dev_inx = bit_ffs(gres_prep->gres_bit_alloc[node_inx]);

	if (dev_inx >= 0) {
		i = -1;
		iter = list_iterator_create(gres_devices);
		while ((gres_device = list_next(iter))) {
			i++;
			if (i == dev_inx) {
				global_id = gres_device->dev_num;
				break;
			}
		}
		list_iterator_destroy(iter);
	}

	/* Epilog: cnt=0 means job ended — clean up per-job cache and return */
	if (!gres_prep->gres_cnt_node_alloc ||
	    !gres_prep->gres_cnt_node_alloc[node_inx]) {
		const char *job_id_str =
			getenvp(*prep_env_ptr, "SLURM_JOB_ID");
		if (job_id_str) {
			if (global_id >= 0) {
				uint32_t job_id = (uint32_t)atol(job_id_str);
				snprintf(buf, sizeof(buf),
					 "%s/hami-job%u-gpu%d.cache",
					 HAMI_CACHE_DIR, job_id, global_id);
				if (unlink(buf) != 0 && errno != ENOENT)
					error("%s: failed to remove cache %s: %m",
					      __func__, buf);
			} else {
				/* best-effort: clean up the gpu-unknown fallback */
				uint32_t job_id = (uint32_t)atol(job_id_str);
				snprintf(buf, sizeof(buf),
					 "%s/hami-job%u-gpu-unknown.cache",
					 HAMI_CACHE_DIR, job_id);
				unlink(buf);
				error("%s: global_id unknown at epilog for job %s;"
				      " cache %s/hami-job%s-gpu*.cache not removed",
				      __func__, job_id_str,
				      HAMI_CACHE_DIR, job_id_str);
			}
		}
		return;
	}

	uint64_t bytes = gres_prep->gres_cnt_node_alloc[node_inx];

	/* LD_PRELOAD */
	_set_ld_preload(prep_env_ptr);

	/* CUDA_DEVICE_MEMORY_LIMIT_0: prolog context uses single-GPU path */
	snprintf(buf, sizeof(buf), "%"PRIu64, bytes);
	env_array_overwrite_fmt(prep_env_ptr, "CUDA_DEVICE_MEMORY_LIMIT_0",
				"%s", buf);

	/* CUDA_DEVICE_MEMORY_SHARED_CACHE: same per-job path as _set_env() */
	{
		const char *job_id_str =
			getenvp(*prep_env_ptr, "SLURM_JOB_ID");
		uint32_t job_id;

		if (job_id_str) {
			job_id = (uint32_t)atol(job_id_str);
		} else {
			error("%s: SLURM_JOB_ID not set; cache isolation disabled",
			      __func__);
			job_id = 0;
		}

		if (global_id >= 0)
			snprintf(buf, sizeof(buf), "%s/hami-job%u-gpu%d.cache",
				 HAMI_CACHE_DIR, job_id, global_id);
		else
			snprintf(buf, sizeof(buf),
				 "%s/hami-job%u-gpu-unknown.cache",
				 HAMI_CACHE_DIR, job_id);
	}

	env_array_overwrite_fmt(prep_env_ptr,
				"CUDA_DEVICE_MEMORY_SHARED_CACHE",
				"%s", buf);
}
