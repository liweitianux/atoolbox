#####################################################################
## XSPEC TCL script
##
## To apply background spectrum correction to MOS1 and MOS2 background.
## It fakes the XB (Galatic + CXB) and NXB (SP + instrument lines)
## spectra according to the current fitting results, and then invoke
## "mathpha" to make the final corrected background spectrum.
##
## NOTE:
## The exposure time for the faked spectra is multiplied by a scale
## factor in order to obtain satisfactory statistics for the higher
## energy regime.  The same scale factor is also applied to the original
## FWC background spectrum, considering that many channels have "counts"
## smaller than one, especially in the higher energy regime.  Otherwise,
## many of these values will be rounded to *zero* by the "mathpha"
## operating in the "COUNT" mode.
## See also the following description to variable "SCALE_FACTOR",
## procedures "spectrum_rate2counts" and "scale_spectrum".
##
##
## NOTES:
## Work with XSPEC v12.x
## Based on "xspec_bkgcorr_v2.tcl" for Chandra background spectrum correction
##
## Weitian LI <liweitianux@gmail.com>
## Created: 2015-11-07
## Updated: 2015-11-12
##
## ChangeLogs:
## 2015-11-10:
##   * Remove "diag_rsp"
##   * Add proc update_header:
##     Update specified keywords instead of just copy the original header
##   * Use lists to record the information for all data groups,
##     therefore this tool can handle any number of data groups.
## 2015-11-11:
##   * Add back "diag_rsp"
## 2015-11-12:
##   * Update to use "ftkeypar" and "fthedit"
##   * Add procedure "scale_spectrum" to scale the FWC background
##   * Add procedure "make_filename"
##   * Add procedure "spectrum_rate2counts" to convert RATE spectrum to
##     COUNTS, therefore this script can properly handle PN spectrum
##   * Update script description
#####################################################################


## Global variables {{{
set datagrp_total [ tcloutr datagrp ]

# Lists to record the information of each data group
set rootname       {}
set rmf            {}
set arf            {}
set diag_rsp       {}
set src_spec       {}
set bkg_spec       {}
set src_exposure   {}
set bkg_exposure   {}
set src_backscale  {}
set bkg_backscale  {}
set modelpars      {}
set back_modelname {}
set back_modelcomp {}
set back_modelpars {}
# This "modelcomp" is not a list
set modelcomp      {}

# FITS header keywords needed to be copyed to the corrected spectrum
set KEYWORDS       {TELESCOP INSTRUME FILTER}

# Scale factor for faking spectra
# By lengthen the exposure time for the fake process, the total counts
# of the faked spectra is much larger, therefore the statistical issuses
# due to small counts (especially in the higher energy regime) is mitigated.
set SCALE_FACTOR   100
## Global variables }}}

## Procedures
# Make new filename by appending the suffix to the stem of filename.
# e.g., "stem.ext" + "sfix" => "stem_sfix.ext"
proc make_filename {filename suffix} {
    regexp -- {(.+)\.([^\.]+)} $filename match stem_ ext_
    set newname "${stem_}_${suffix}.${ext_}"
    return $newname
}

# Determine root name and get other filenames (e.g., rmf, arf, back)
proc get_filenames {datagrp} {
    global rootname
    global src_spec
    global bkg_spec
    global rmf
    global arf
    global diag_rsp
    set src_spec_ [ exec basename [ tcloutr filename $datagrp ] ]
    regexp -- {(.+)_(bin|grp|group)([0-9]*)\.(pi|fits)} $src_spec_ match rootname_ grptype_ grpval_ fext_
    set bkg_spec_ [ tcloutr backgrnd $datagrp ]
    set response_ [ tcloutr response $datagrp ]
    set rmf_      [ lindex $response_ 0 ]
    set diag_rsp_ [ lindex $response_ 1 ]
    set arf_      [ lindex [ tcloutr arf      $datagrp ] 0 ]
    lappend rootname $rootname_
    lappend src_spec $src_spec_
    lappend bkg_spec $bkg_spec_
    lappend rmf      $rmf_
    lappend arf      $arf_
    lappend diag_rsp $diag_rsp_
}

# Get exposure and backscale values
proc get_exposure_backscale {datagrp} {
    global src_exposure
    global bkg_exposure
    global src_backscale
    global bkg_backscale
    set src_exposure_ [ tcloutr expos $datagrp s ]
    set bkg_exposure_ [ tcloutr expos $datagrp b ]
    scan [ tcloutr backscal $datagrp s ] "%f" src_backscale_
    scan [ tcloutr backscal $datagrp b ] "%f" bkg_backscale_
    lappend src_exposure  $src_exposure_
    lappend bkg_exposure  $bkg_exposure_
    lappend src_backscale $src_backscale_
    lappend bkg_backscale $bkg_backscale_
}

# Get the model components
proc get_modelcomp {} {
    global datagrp_total
    global modelcomp
    # NOTE: "model" contains all the model components corresponding to
    # echo data group, and contains also the background model components.
    # Therefore, the "model" contents repeats the same model components
    # for each data group.
    # e.g.,
    # if data group 1 has model components: "wabs*apec", and there are
    # altogether 2 data groups, then the "model" content is:
    #     "wabs(apec) wabs(apec)"
    # which is repeated!
    set model [ tcloutr model ]
    set indices [ lsearch -all $model {*:} ]
    if { [ llength $indices ] == 0 } {
        set modelcomp $model
    } else {
        set modelcomp [ lrange $model 0 [ expr {[ lindex $indices 0 ] - 1} ] ]
    }
    # Deal with the repeated model components if exist multiple data groups
    set comp_length [ expr { [ llength $modelcomp ] / $datagrp_total } ]
    set modelcomp [ lrange $modelcomp 0 [ expr {$comp_length - 1} ] ]
}

# Get the model parameters
proc get_modelpars {datagrp} {
    global datagrp_total
    global modelpars
    set modelpars_ ""
    # NOTE: "modpar" is the total number of parameters,
    # not just for one data group
    set npar_ [ expr { [ tcloutr modpar ] / $datagrp_total } ]
    set par_begin_ [ expr {$npar_ * ($datagrp - 1) + 1} ]
    set par_end_   [ expr {$npar_ + $par_begin_ - 1} ]
    for {set i $par_begin_} {$i <= $par_end_} {incr i} {
        scan [ tcloutr param $i ] "%f" pval_
        set modelpars_ "$modelpars_ $pval_ &"
    }
    lappend modelpars $modelpars_
}

# Get the model name, components and parameters of the background model
# corresponding to the given data group
proc get_backmodel {datagrp} {
    global back_modelname
    global back_modelcomp
    global back_modelpars
    set model_ [ tcloutr model ]
    set indices_ [ lsearch -all $model_ {*:} ]
    set name_idx_ [ lindex $indices_ [ expr {$datagrp - 1} ] ]
    set back_modelname_ [ regsub {:\s*} [ lindex $model_ $name_idx_ ] "" ]
    set ncomp_ [ tcloutr modcomp $back_modelname_ ]
    set back_modelcomp_ [ lrange $model_ [ expr {$name_idx_ + 1} ] [ expr {$name_idx_ + $ncomp_ } ] ]
    set npar_ [ tcloutr modpar $back_modelname_ ]
    set back_modelpars_ ""
    for {set i 1} {$i <= $npar_} {incr i} {
        scan [ tcloutr param ${back_modelname_}:$i ] "%f" pval_
        set back_modelpars_ "$back_modelpars_ $pval_ &"
    }
    lappend back_modelname $back_modelname_
    lappend back_modelcomp $back_modelcomp_
    lappend back_modelpars $back_modelpars_
}

# Save current XSPEC fitting results
proc save_xspec_fit {} {
    set now [ exec date +%Y%m%d%H%M ]
    set xspec_outfile "xspec_${now}.xcm"
    if { [ file exists ${xspec_outfile} ] } {
        exec mv -fv ${xspec_outfile} ${xspec_outfile}_bak
    }
    save all "${xspec_outfile}"
}

# Load model
proc load_model {comp pars} {
    model clear
    model $comp & $pars /*
}

# Set the norm of the ICM APEC component to be zero,
# which is the last parameter of current loaded model.
proc set_icm_norm {} {
    set npar_ [ tcloutr modpar ]
    newpar $npar_ 0.0
    puts "ICM_apec_norm:$npar_: [ tcloutr param $npar_ ]"
}

# Fake spectrum
proc fake_spectrum {outfile rmf arf exptime} {
    if {[ file exists ${outfile} ]} {
        exec mv -fv ${outfile} ${outfile}_bak
    }
    data none
    fakeit none & $rmf & $arf & y &  & $outfile & $exptime & /*
}

# Convert a spectrum of "RATE" to a spectrum of "COUNTS" by multiplying
# the exposure time.  The "STAT_ERR" column is also scaled if exists.
# NOTE:
# The MOS1/2 FWC background spectrum generated by "mos_back" has column
# "COUNTS", however, the PN FWC background spectrum generated by "pn_back"
# has column "RATE".
# Therefore, the PN background spectra should be converted to "COUNTS" format,
# since the other faked CXB & NXB spectra are also in "COUNTS" format.
# The exposure time is acquired from the "EXPOSURE" keyword in the header
# of input spectrum.
#
# Return:
#   * 0  : input spectrum is already in "COUNTS" format, no conversion needed
#   * 1  : input spectrum is in "RATE" format, and is converted to "COUNTS"
#          format and write to outfile
#   * -1 : invalid input spectrum
proc spectrum_rate2counts {outfile infile} {
    if {[ file exists ${outfile} ]} {
        exec mv -fv ${outfile} ${outfile}_bak
    }
    # Get number of columns/fields
    exec ftkeypar "${infile}+1" TFIELDS
    set fields_ [ exec pget ftkeypar value ]
    set colnames {}
    for {set i 1} {$i <= $fields_} {incr i} {
        exec ftkeypar "${infile}+1" "TTYPE${i}"
        set val_ [ exec pget ftkeypar value ]
        set colname_ [ string toupper [ string trim $val_ {' } ] ]
        lappend colnames $colname_
    }
    if { [ lsearch $colnames {COUNTS} ] != -1 } {
        # Input spectrum is already in "COUNTS" format
        return 0
    } elseif { [ lsearch $colnames {RATE} ] != -1 } {
        # Get exposure time
        exec ftkeypar "${infile}+1" EXPOSURE
        set exposure_ [ exec pget ftkeypar value ]
        # Convert column "RATE"
        set tmpfile "${outfile}_tmp[pid]"
        exec cp -fv ${infile} ${tmpfile}
        # COUNTS = RATE * EXPOSURE
        set fcmd_ "ftcalc \"${tmpfile}\" \"${outfile}\" column=RATE expression=\"RATE * $exposure_\" history=yes"
        puts $fcmd_
        exec sh -c $fcmd_
        exec rm -fv ${tmpfile}
        # Update column name from "RATE" to "COUNTS", and column units
        set field_idx_ [ expr { [ lsearch $colnames {RATE} ] + 1 } ]
        set fcmd_ "fthedit \"${outfile}+1\" keyword=\"TTYPE${field_idx_}\" operation=add value=\"COUNTS\""
        puts $fcmd_
        exec sh -c $fcmd_
        set fcmd_ "fthedit \"${outfile}+1\" keyword=\"TUNIT${field_idx_}\" operation=add value=\"count\""
        puts $fcmd_
        exec sh -c $fcmd_
        # Convert column "STAT_ERR"
        if { [ lsearch $colnames {STAT_ERR} ] != -1 } {
            exec mv -fv ${outfile} ${tmpfile}
            # STAT_ERR = STAT_ERR * EXPOSURE
            set fcmd_ "ftcalc \"${tmpfile}\" \"${outfile}\" column=STAT_ERR expression=\"STAT_ERR * $exposure_\" history=yes"
            puts $fcmd_
            exec sh -c $fcmd_
            exec rm -fv ${tmpfile}
        }
        return 1
    } else {
        # Invalid input spectrum
        return -1
    }
}

# Scale the spectrum "COUNTS" and "STAT_ERR" (if exists) columns by a factor.
# NOTE:
# This procedure is mainly used to scale the FWC background, within which
# many channels have "counts" less than one.  When using "mathpha" to combine
# this FWC background and other faked spectra in "COUNT" mode, the channels
# with "counts" less than one (especially the higher energy regime) may be
# truncated/rounded to integer, therefore the higher energy regime of the
# combined spectrum may be greatly underestimated, and will cause the
# following fitting very bad in the higher energy regime.
# See also the description about the "SCALE_FACTOR" above.
proc scale_spectrum {outfile infile factor} {
    if {[ file exists ${outfile} ]} {
        exec mv -fv ${outfile} ${outfile}_bak
    }
    exec cp -fv ${infile} ${outfile}
    # Get number of columns/fields
    exec ftkeypar "${outfile}+1" TFIELDS
    set fields_ [ exec pget ftkeypar value ]
    for {set i 1} {$i <= $fields_} {incr i} {
        exec ftkeypar "${outfile}+1" "TTYPE${i}"
        set val_ [ exec pget ftkeypar value ]
        set colname_ [ string toupper [ string trim $val_ {' } ] ]
        # Scale column "COUNTS"
        if {$colname_ == "COUNTS"} {
            set tmpfile "${outfile}_tmp[pid]"
            exec mv -fv ${outfile} ${tmpfile}
            # COUNTS = COUNTS * factor
            set fcmd_ "ftcalc \"${tmpfile}\" \"${outfile}\" column=COUNTS expression=\"COUNTS * $factor\" history=yes"
            puts $fcmd_
            exec sh -c $fcmd_
            exec rm -fv ${tmpfile}
        }
        # Scale column "STAT_ERR"
        if {$colname_ == "STAT_ERR"} {
            set tmpfile "${outfile}_tmp[pid]"
            exec mv -fv ${outfile} ${tmpfile}
            # STAT_ERR = STAT_ERR * factor
            set fcmd_ "ftcalc \"${tmpfile}\" \"${outfile}\" column=STAT_ERR expression=\"STAT_ERR * $factor\" history=yes"
            puts $fcmd_
            exec sh -c $fcmd_
            exec rm -fv ${tmpfile}
        }
    }
}

# Combine faked spectra to original FWC background spectrum with "mathpha"
proc combine_spectra {outfile fwc_bkg cxb nxb exposure backscale} {
    set combine_expr_ "$fwc_bkg + $cxb + $nxb"
    set comment1_ "Corrected background spectrum; FWC based"
    set combine_cmd_ "mathpha expr='${combine_expr_}' outfil=${outfile} exposure=${exposure} backscal=${backscale} units=C areascal=% properr=yes ncomments=1 comment1='${comment1_}'"
    if {[ file exists ${outfile} ]} {
        exec mv -fv ${outfile} ${outfile}_bak
    }
    puts $combine_cmd_
    exec sh -c $combine_cmd_
}

# Copy header keywords from original background spectrum to the
# newly combined background spectrum
proc update_header {newfile origfile} {
    global KEYWORDS
    for {set i 0} {$i < [ llength $KEYWORDS ]} {incr i} {
        set key_ [ lindex $KEYWORDS $i ]
        exec ftkeypar "${origfile}+1" $key_
        set val_ [ exec pget ftkeypar value ]
        #set fcmd_ "fparkey value=\"$val_\" fitsfile=\"${newfile}+1\" keyword=\"$key_\" add=yes"
        set fcmd_ "fthedit \"${newfile}+1\" keyword=\"$key_\" operation=add value=\"$val_\""
        puts $fcmd_
        exec sh -c $fcmd_
    }
}


# Save current fit results
save_xspec_fit

# Get the current model components, which is the same for all data groups
get_modelcomp

for {set dg 1} {$dg <= $datagrp_total} {incr dg} {
    get_filenames          $dg
    get_exposure_backscale $dg
    get_modelpars          $dg
    get_backmodel          $dg
}

puts "modelcomp: $modelcomp"
puts "rootname: $rootname"
puts "rmf: $rmf"
puts "arf: $arf"
puts "diag_rsp: $diag_rsp"
puts "src_spec: $src_spec"
puts "bkg_spec: $bkg_spec"
puts "src_exposure: $src_exposure"
puts "bkg_exposure: $bkg_exposure"
puts "src_backscale: $src_backscale"
puts "bkg_backscale: $bkg_backscale"
puts "modelpars: $modelpars"
puts "back_modelname: $back_modelname"
puts "back_modelcomp: $back_modelcomp"
puts "back_modelpars: $back_modelpars"
puts "SCALE_FACTOR: $SCALE_FACTOR"

# DEBUG
set DEBUG 0
if {$DEBUG != 1} {

# Clear all loaded model data and models
data none
model clear

for {set idg 0} {$idg < $datagrp_total} {incr idg} {
    set rootname_       [ lindex $rootname       $idg ]
    set bkg_spec_       [ lindex $bkg_spec       $idg ]
    set rmf_            [ lindex $rmf            $idg ]
    set arf_            [ lindex $arf            $idg ]
    set diag_rsp_       [ lindex $diag_rsp       $idg ]
    set bkg_exposure_   [ lindex $bkg_exposure   $idg ]
    set bkg_backscale_  [ lindex $bkg_backscale  $idg ]
    set modelpars_      [ lindex $modelpars      $idg ]
    set back_modelcomp_ [ lindex $back_modelcomp $idg ]
    set back_modelpars_ [ lindex $back_modelpars $idg ]
    set fake_exposure   [ expr {$bkg_exposure_  * $SCALE_FACTOR} ]
    set fake_backscale  [ expr {$bkg_backscale_ * $SCALE_FACTOR} ]
    set fake_cxb        "fake_cxb_${rootname_}.pi"
    set fake_nxb        "fake_nxb_${rootname_}.pi"
    set bkgspec_cnts    [ make_filename $bkg_spec_ "cnts" ]
    set bkgcorr_outfile "bkgcorr_${rootname_}.pi"
    # Fake CXB and NXB
    load_model    $modelcomp $modelpars_
    set_icm_norm
    fake_spectrum $fake_cxb $rmf_ $arf_ $fake_exposure
    load_model    $back_modelcomp_ $back_modelpars_
    fake_spectrum $fake_nxb $diag_rsp_ "" $fake_exposure
    # Convert FWC background spectrum from "RATE" to "COUNTS" for PN
    set ret [ spectrum_rate2counts $bkgspec_cnts $bkg_spec_ ]
    if { $ret == 0 } {
        set bkgspec_cnts $bkg_spec_
    } elseif { $ret == 1 } {
        puts "Converted RATE spectrum to COUNTS: $bkgspec_cnts"
    } else {
        return -code error "ERROR: invalid spectrum for 'spectrum_rate2counts'"
    }
    set bkgspec_scaled [ make_filename $bkgspec_cnts "sf${SCALE_FACTOR}" ]
    # Scale original FWC background before combination
    scale_spectrum  $bkgspec_scaled $bkgspec_cnts $SCALE_FACTOR
    # Combine faked spectra with original FWC background
    combine_spectra $bkgcorr_outfile $bkgspec_scaled $fake_cxb $fake_nxb $bkg_exposure_ $fake_backscale
    update_header   $bkgcorr_outfile $bkg_spec_
}

}

# vim: set ts=8 sw=4 tw=0 fenc=utf-8 ft=tcl: #
