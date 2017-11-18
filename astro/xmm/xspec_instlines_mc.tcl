#
# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT License
#
# XSPEC (v12) TCL script
#
# Description
# -----------
# This script employs the Monte Carlo method to evaluate the impacts of
# the instrumental lines (modeled as Gaussians) on the fitting results.
# For each iteration, the norms of the Gaussians are randomly altered
# according to the its fitted value and sigma, and fixed, and then fit
# the model to obtain the new result. All Monte Carlo results are saved
# to a text file for later analyses, in order to evaluate the systematic
# impacts of the instrumental lines.
#
# Usage
# -----
# 1. already fitted spectral model (proper parameters free/thaw/link etc.)
# 2. "set mc_times <mc-times>"; number of Monte Carlo times (default: 1000)
# 3. "set outfile <outfile>"; output file to save the results
# 4. @<this-script>
#
# References
# ----------
# * XSPEC - tclout, tcloutr
#   https://heasarc.gsfc.nasa.gov/docs/xanadu/xspec/manual/XStclout.html
#   https://heasarc.gsfc.nasa.gov/docs/xanadu/xspec/manual/XStcloutr.html
# * XSPEC - The User Interface
#   https://heasarc.gsfc.nasa.gov/docs/xanadu/xspec/manual/XSappendixTcl.html
# * TCL 8.5 - string
#   https://www.tcl.tk/man/tcl8.5/TclCmd/string.htm
#
# 2017-11-17
#

#
# XSPEC control
#

# Do not echo commands
set xs_echo_script 0
# Renormalize only at the beginning of a fit
renorm prefit
# Do not ask whether to continue, just continue
set query [ tcloutr query ]
query yes
# Do not chatter too much
set chatlevel [ lindex [ tcloutr chatter ] 0 ]
chatter 5

#
# Settings
#

# Name of this script
set script_name "xspec_instlines_mc.tcl"
set time_now [ clock format [ clock seconds ] -format %Y%m%dT%H%M ]
# Default output filename to store the Monte Carlo results
set outfile_default "xspec_instlines_mc.${time_now}.csv"
# Default file to log the XSPEC messages (as the screen output is suppressed)
set logfile_default "xspec_instlines_mc.${time_now}.log"
# Default number of Monte Carlo times
set mc_times_default 1000

#
# Procedures
#

# Save current XSPEC fitting results
proc save_xspec {} {
    global time_now
    set xspec_outfile "xspec_saveall.${time_now}.xcm"
    save all $xspec_outfile
    puts "XSPEC fitting results saved to: $xspec_outfile"
}

# Print the header line of the given parameters
proc print_header {parameters {fd stdout} {sep ,}} {
    set plabels {}
    foreach p $parameters {
        set dg_ [ dict get $p datagrp ]
        set cname_ [ dict get $p comp_name ]
        set cnum_ [ dict get $p comp_num ]
        set pname_ [ dict get $p par_name ]
        set pnum_ [ dict get $p par_num ]
        set label "${dg_}/${cname_}.${cnum_}/${pname_}(${pnum_})"
        lappend plabels $label
    }
    puts $fd [ join $plabels $sep ]
}

# Print the data line of the given parameters
proc print_data {parameters {fd stdout} {sep ,}} {
    set pvalues {}
    foreach p $parameters {
        set pnum_ [ dict get $p par_num ]
        set pval_ [ lindex [ tcloutr param $pnum_ ] 0 ]
        lappend pvalues $pval_
    }
    puts $fd [ join $pvalues $sep ]
}

# Generate normal distribution random number
# Box-Mueller method
proc prng_normal {{mean 0.0} {stdev 1.0}} {
    set pi 3.141592653589793
    set rad_ [ expr {sqrt(-2.8 * log(rand()))} ]
    set phi_ [ expr {2.0 * $pi * rand()} ]
    set r_ [ expr {$rad_ * cos($phi_)} ]
    return [ expr {$mean + $stdev * $r_} ]
}

# Randomize the norms of the Gaussian models according to their fitted
# values and sigmas
proc randomize_norms {parameters} {
    foreach p $parameters {
        set pnum_ [ dict get $p norm_par ]
        set value_ [ dict get $p norm_value ]
        set sigma_ [ dict get $p norm_sigma ]
        if {$sigma_ > 0.0} {
            set newval_ [ prng_normal $value_ $sigma_ ]
            if {$newval_ < 0.0} {
                set newval_ 0.0
            }
            newpar $pnum_ $newval_
        }
    }
}

# Freeze the norms of the Gaussians models
proc freeze_norms {parameters} {
    foreach p $parameters {
        set pnum_ [ dict get $p norm_par ]
        set pdelta_ [ lindex [ tcloutr param $pnum_ ] 1 ]
        set plink_ [ lindex [ tcloutr plink $pnum_ ] 0 ]
        if {$pdelta_ > 0.0 && $plink_ == "F"} {
            freeze $pnum_
        }
    }
}

#
# Main
#

save_xspec

# Output log file
if {[ info exists logfile ]} {
    puts "Output results file: ${logfile}"
} else {
    puts "WARNING: logfile not set, default to ${logfile_default}"
    set logfile $logfile_default
}
if {[ file exists $logfile ]} {
    file rename -force $logfile ${logfile}.old
}
# Enable log to file
log $logfile

# Output results file
if {[ info exists outfile ]} {
    puts "Output results file: ${outfile}"
} else {
    puts "WARNING: outfile not set, default to ${outfile_default}"
    set outfile $outfile_default
}
if {[ file exists $outfile ]} {
    file rename -force $outfile ${outfile}.old
}
set outfd [ open $outfile w ]

if {[ info exists mc_times ]} {
    puts "Number of Monte Carlo iterations: ${mc_times}"
} else {
    puts "WARNING: mc_times not set, default to ${mc_times_default} times"
    set mc_times $mc_times_default
}

set ndatagrp [ tcloutr datagrp ]
set nmodcomp [ tcloutr modcomp ]
puts "Number of data groups: ${ndatagrp}"
puts "Number of model components: ${nmodcomp}"

# List of Gaussian lines, of which each element is an dictionary of elements:
#   - datagrp : data group number
#   - comp : model component number
#   - norm_par : parameter number of the Gaussian norm
#   - norm_value : Gaussian norm value
#   - norm_sigma : Gaussian norm sigma (i.e., standard deviation)
set gaussians {}
puts "-----------------------------------------------------------------------"
for {set i 1} {$i <= $ndatagrp} {incr i} {
    for {set j 1} {$j <= $nmodcomp} {incr j} {
        set comp_ [ tcloutr compinfo $j $i ]
        set comp_name_ [ lindex $comp_ 0 ]
        if {[ string equal $comp_name_ gaussian ]} {
            set norm_par_ [ expr {[ lindex $comp_ 1 ] + 2}]
            set norm_value_ [ lindex [ tcloutr param $norm_par_ ] 0 ]
            set norm_sigma_ [ tcloutr sigma $norm_par_ ]
            set gaus_ [ dict create \
                            datagrp $i \
                            comp $j \
                            norm_par $norm_par_ \
                            norm_value $norm_value_ \
                            norm_sigma $norm_sigma_ ]
            lappend gaussians $gaus_
            puts "${i}/gaussian: ${gaus_}"
        }
    }
}
puts "Number of Gaussian lines: [ llength $gaussians ]"

# List of free parameters, of which each element is an dictionary of elements:
#   - datagrp : data group number
#   - comp_num : model component number
#   - comp_name : model component name
#   - par_num : parameter number
#   - par_name : parameter name
set freeparameters {}
puts "-----------------------------------------------------------------------"
for {set i 1} {$i <= $ndatagrp} {incr i} {
    for {set j 1} {$j <= $nmodcomp} {incr j} {
        set comp_ [ tcloutr compinfo $j $i ]
        set comp_name_ [ lindex $comp_ 0 ]
        set comp_pfirst_ [ lindex $comp_ 1 ]  ;# 1st parameter number
        set comp_np_ [ lindex $comp_ 2 ]  ;# number of parameters
        set comp_plast_ [ expr {$comp_pfirst_ + $comp_np_ - 1} ]
        for {set k $comp_pfirst_} {$k <= $comp_plast_} {incr k} {
            scan [ tcloutr param $k ] "%f %f" par_value_ par_delta_
            set par_link_ [ lindex [ tcloutr plink $k ] 0 ]
            set par_sigma_ [ tcloutr sigma $k ]
            if {$par_delta_ > 0.0 && $par_link_ == "F"} {
                set par_name_ [ lindex [ tcloutr pinfo $k ] 0 ]
                set param_ [ dict create \
                                 datagrp $i \
                                 comp_num $j \
                                 comp_name $comp_name_ \
                                 par_num $k \
                                 par_name $par_name_ ]
                lappend freeparameters $param_
                puts "${i}/${comp_name_}.${j}/${par_name_}(${k}):\
                      ${par_value_} +/- ${par_sigma_}"
            }
        }
    }
}
puts "Number of free parameters: [ llength $freeparameters ]"
print_header $freeparameters $outfd

puts "-----------------------------------------------------------------------"
freeze_norms $gaussians
set tstart [ clock seconds ]
for {set i 0} {$i < $mc_times} {incr i} {
    puts -nonewline "... [ expr {$i + 1} ] / ${mc_times} ..."
    if {$i == 0} {
        puts ""
    } else {
        set tnow [ clock seconds ]
        set elapsed [ expr {($tnow - $tstart) / 60.0} ]
        set eta [ expr {$elapsed * ($mc_times - $i) / $i} ]
        puts [ format " Elapsed %.1f min / ETA %.1f min ..." $elapsed $eta ]
    }
    randomize_norms $gaussians
    fit
    print_data $freeparameters $outfd
}
puts "-----------------------------------------------------------------------"
set tnow [ clock seconds ]
set elapsed [ expr {($tnow - $tstart) / 60.0} ]
puts [ format "Total Monte Carlo time: %.1f min" $elapsed ]

close $outfd
# Recover query and chatter level
query $query
chatter $chatlevel
# Disable log to file
log none

puts "Check the log file for more details: ${logfile}"
puts "Results written into: ${outfile}"
