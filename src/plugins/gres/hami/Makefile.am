# Makefile for gres/hami plugin

AUTOMAKE_OPTIONS = foreign

PLUGIN_FLAGS = -module -avoid-version --export-dynamic

AM_CPPFLAGS = -DSLURM_PLUGIN_DEBUG -I$(top_srcdir) $(NVML_CPPFLAGS)

pkglib_LTLIBRARIES = gres_hami.la

# Gres HAMi plugin.
gres_hami_la_SOURCES = gres_hami.c
gres_hami_la_LDFLAGS = $(PLUGIN_FLAGS)
gres_hami_la_LIBADD  = ../common/libgres_common.la

force:
$(gres_hami_la_LIBADD) : force
	@cd `dirname $@` && $(MAKE) `basename $@`
