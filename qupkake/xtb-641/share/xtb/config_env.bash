#!/bin/bash
# run this script to set up a xtb environment
# requirements: $XTBHOME is set to `pwd`
if [ -z "${XTBHOME}" ]; then
   XTBHOME="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../"
fi

# set up path for xtb, using the xtb directory and the users home directory
XTBPATH=${XTBHOME}/share/xtb:${HOME}

# to include the documentation we include our man pages in the users manpath
MANPATH=${MANPATH}:${XTBHOME}/share/man

# finally we have to make the binaries
PATH=${PATH}:${XTBHOME}/bin

# enable package config for xtb
PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${XTBHOME}/lib64/pkgconfig

export PATH XTBPATH MANPATH PKG_CONFIG_PATH
