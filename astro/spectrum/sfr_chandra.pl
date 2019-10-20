#!/usr/bin/env perl
#
# Copyright (c) 2015 Weitian LI <liweitianux@gmail.com>
# Copyright (c) 2007 Junhua GU <tompkins@sjtu.edu.cn>
#
# MIT License
#
# Performs automatically spectral fitting with XSPEC 12.
# For Chandra data only.
#
# Reference:
# [1] Running Multiple Copies of a Tool - CIAO
#     http://cxc.harvard.edu/ciao/ahelp/parameter.html#Running_Multiple_Copies_of_a_Tool
#

use strict;
use warnings;
use File::Basename;
use File::Copy;

my $version = "2019/10/16";
#
# ChangeLogs:
#
# 2019/10/16: Weitian LI
#   * support region list file with each line representing a region file
#     instead of the region itself (Sanders's contour binning
#     segmentation program output such a region format:
#     http://www-xray.ast.cam.ac.uk/papers/contbin/ )
#
# 2015/03/04: Weitian LI
#   Completely rewrite of this script:
#   * add use strict, warnings, many fixes to the syntax
#   * add sub 'usage'
#   * rewrite sub 'update_backscale' -> 'rescale'
#   * update sub 'extract_spec' -> 'extract_spectrum'
#   * remove sub 'extract_back'
#   * replace 'grppha' with 'dmgroup'
#   * add sub 'determine_bkg' to determine the type of background
#   * fix the problem of when background set to 'none'
#   * copy CIAO tools' parameter files to local, avoiding conflicts when
#     running multiple instances of a CIAO tool
#   * allow comments in fit config file (starting with '//')
#   * update sub 'parse_result' to accept the parameter list and hash,
#     for better/simple parse process
#   * update example fit config file
#   * update generation of xspec fit script
#   * update config file syntax: merged '#PARA' and '#ERROR'
#   * use backtics to get the output of command within perl
#   * !! INCOMPATIBLE !! improve/change output format (columns adjustments)
#
# 2015/02/05: Weitian LI
#   * fix the wrong use of chomp.
#
# 2015/01/27: Weitian LI
#   * check 'cnt_spec' value before adjust the backscale
#
# 2015/01/21:
#   * add 'query yes' to temp_fit_script
#

## usage {{{
sub usage {
    my $prog = basename($0);
    my $text = qq(SFR - Spectral Fitting Robot
Usage:
    $prog <region_list> <evt> <bkgd> <output> <rmf> <arf> <fit_config>

Options:
    region_list: each line can be a region string or a region file

Version: $version
);
    print $text;
}
## usage }}}

# Example fit config {{{
#----------------------------------------------------------------
#    abund grsa
#    weight gehrels
#
#    model wabs*apec & <nh> & 1.0 & 0.5 & <redshift> & 1.0 &
#
#    freeze 1
#    thaw 3
#
#    ignore **:0.0-0.7,7.0-**
#
#    fit
#    fit
#    fit
#
#    // Do not adjust BACKSCAL of background spectrum
#    #NORESCALE
#
#    #PARA kT    2
#
#    // Also calculate the errors of kT with XSPEC error command
#    #PARA kT    2 ERROR 1.0
#    #PARA Abund 3
#    #STAT
#    #FLUX 0.7 7.0
#
#    // Specify the parameters for dmgroup
#    #DMGROUP NUM_CTS 20
#    #DMGROUP BIN     1:128:2,129:256:4,257:512:8,513:1024:16
#----------------------------------------------------------------
# example config }}}

## variables {{{
# Background types
use constant {
    NO_BKG   => 0,  # do not use a background
    BLANKSKY => 1,  # blanksky fits file
    STOW_BKG => 2,  # stowed background
    LBKG_REG => 3,  # region file of a local background
    BKG_SPEC => 4,  # background spectrum file
};

# Threshold value for the 'reduced chi-squared'
my $rechisq_threshold = 1.4;
# Number of bad fitting (If fitted rechisq > $rechisq_threshold)
my $bad_fit_num = 0;

# Whether to rescale the background (default yes)
my $rescale = 1;

# Energy range selected to adjust the BACKSCAL of the background spectra.
# PI = [ energy(eV) / 14.6 eV + 1 ]
# Chandra particle background energy range:
# 9.5-12.0 keV (channel: 651-822)
my $pb_chn_lo = 651;
my $pb_chn_hi = 822;

# Source spectrum particle background counts ($cnt_spec,
# within channel back_lo -> back_hi) may be too small, even ZERO,
# thus cause error of zero division.
# If '$cnt_spec' too small (say 20), then skip adjusting the BACKSCAL.
my $pb_cnt_threshold = 20;

# 'dmgroup' parameters
my $grouptype = "NUM_CTS";
#my $grouptype="BIN"
my $grouptypeval = "20";
my $binspec = "1:128:2,129:256:4,257:512:8,513:1024:16";
## variables }}}

# NOTE: @ARGV does NOT include $0, the name of the program.
#my $argc = @ARGV;
my $argc = $#ARGV + 1;
#print "argc: $argc\n";
if ($argc != 7) {
    usage();
    exit 1;
}

## Setup local PFILES environment {{{
# Allow running multiple copies of a CIAO tool.
# Reference: http://cxc.harvard.edu/ciao/ahelp/parameter.html#Running_Multiple_Copies_of_a_Tool
#
# Make a local copy of necessary parameter files.
print "Setup local minimal environment for CIAO to avoid pfile conflicts.\n";
print "Copy necessary parameter files ...\n";
my @ciao_tools = ("dmstat", "dmkeypar", "dmhedit", "dmextract", "dmgroup");
foreach (@ciao_tools) {
    my $tool = $_;
    my $tool_par = `paccess $tool`;
    chomp($tool_par);
    #print "${tool}.par: $tool_par.\n";
    system("punlearn $tool && cp -Lv $tool_par .");
}

# Set environment variable PFILES
$ENV{PFILES} = "./:$ENV{PFILES}";
#system("echo PFILES: \$PFILES");
## PFILES }}}

## process arguments {{{
# File contains the list of fitting regions
my $region_list_fn = $ARGV[0];
# Event file (clean)
my $evt2_fn = $ARGV[1];
# Background file (blanksky.fits / bkg.pi)
my $bkg_fn = $ARGV[2];
# Output file to save fitting results
my $output_fn = $ARGV[3];
my $rmf_fn = $ARGV[4];
my $arf_fn = $ARGV[5];
# Fit config file for this tool
my $config_fn = $ARGV[6];

print "====================================================\n";
print "region_list: $region_list_fn\n";
print "evt2:        $evt2_fn\n";
print "bkg:         $bkg_fn \n";
print "output:      $output_fn\n";
print "rmf:         $rmf_fn\n";
print "arf:         $arf_fn\n";
print "config:      $config_fn\n";

# Parse evt, back, rmf, arf ... {{{
my @evt_list = split(/[,\ ]+/, $evt2_fn);
my $evt_num  = @evt_list;
my @bkg_list = split(/[,\ ]+/, $bkg_fn);
my $bkg_num  = @bkg_list;
my @rmf_list = split(/[,\ ]+/, $rmf_fn);
my $rmf_num  = @rmf_list;
my @arf_list = split(/[,\ ]+/, $arf_fn);
my $arf_num  = @arf_list;

print "====================================================\n";
print "evt_num: $evt_num, bkg_num: $bkg_num, rmf_num: $rmf_num, arf_num: $arf_num\n";
if (! (($evt_num == $bkg_num) && ($bkg_num == $rmf_num) && ($rmf_num == $arf_num))) {
    print "ERROR: number of evt, bkg, rmf or arf NOT match!\n";
    exit 12;
}

foreach my $i (0 .. $#evt_list) {
    printf "Data group %d: ", ($i+1);
    printf "%s, %s, %s, %s\n", $evt_list[$i], $bkg_list[$i], $rmf_list[$i], $arf_list[$i];
}
print "====================================================\n";
# parse evt, bkg ... }}}

# Backup the output file if it exists
if ( -e $output_fn ) {
    print "WARNING: Output file '$output_fn' already exists!\n";
    system("mv -fv $output_fn ${output_fn}_bak");
}

# Determine the type of each background file
my @bkg_type = ();
foreach my $bkgd (@bkg_list) {
    my $bkgd_type = NO_BKG;
    if ($bkgd =~ /^(no|null|none)$/i) {
        print "WARNING: does not use background!\n";
        $bkgd_type = NO_BKG;
    } elsif (! -e $bkgd) {
        print "ERROR: bkg file '$bkgd' does not exists!\n";
        exit 11;
    } else {
        $bkgd_type = determine_bkg($bkgd);
    }
    push(@bkg_type, $bkgd_type);
}
## arguments }}}

## Fitting related variables {{{
my $fit_reg_fn    = "_fit.reg";
my $fit_script_fn = "_fit.tcl";
my $fit_result_fn = "_fit.txt";
if (-e $fit_result_fn) {
    unlink($fit_result_fn);
}

my @fit_spec_list = ();
my @fit_grp_spec_list = ();
my @fit_bkg_spec_list = ();
foreach my $i (0 .. $#evt_list) {
    my $ii = $i + 1;
    push (@fit_spec_list,     "_fit_${ii}.pi");
    push (@fit_grp_spec_list, "_fit_${ii}_grp.pi");
    push (@fit_bkg_spec_list, "_fit_${ii}_bkg.pi");
}
## varaibles }}}

## Subroutines {{{
# Determine the type of the given background file {{{
# Usage: determine_bkg($bkg_filename)
sub determine_bkg {
    my $argc = @_;
    if ($argc != 1) {
        print "ERROR: sub determine_bkg() requires 1 argument!\n";
        exit 21;
    }
    my $bkgd = $_[0];
    my $filetype = `file -bL $bkgd`;
    if ($filetype =~ /^ASCII text/) {
        print "determine_bkg: '$bkgd' is ASCII text\n";
        return LBKG_REG;
    } elsif ($filetype =~ /^FITS/) {
        print "determine_bkg: '$bkgd' is FITS\n";
        my $hduclas1 = `punlearn dmkeypar && dmkeypar $bkgd HDUCLAS1 echo=yes`;
        chomp($hduclas1);
        if ($hduclas1 =~ /^EVENTS$/) {
            print "determine_bkg: '$bkgd' is FITS of EVENTS\n";
            my $bkg_obj = `punlearn dmkeypar && dmkeypar $bkgd OBJECT echo=yes`;
            chomp($bkg_obj);
            if ($bkg_obj =~ /^BACKGROUND DATASET$/) {
                print "determine_bkg: '$bkgd' is background dataset\n";
                return BLANKSKY;
            } elsif ($bkg_obj =~ /^ACIS STOWED$/) {
                print "determine_bkg: '$bkgd' is stowed background\n";
                return STOW_BKG;
            } else {
                print "ERROR: determine_bkg: unknown EVENTS type\n";
                exit 22;
            }
        } elsif ($hduclas1 =~ /^SPECTRUM$/) {
            print "determine_bkg: '$bkgd' is FITS of SPECTRUM\n";
            return BKG_SPEC;
        } else {
            print "ERROR: determine_bkg: unknown FITS type\n";
            exit 23;
        }
    } else {
        print "ERROR: determine_bkg: unknown file type: $filetype\n";
        exit 24;
    }
}
# determine_bkg }}}

# Adjust BACKSCALE of background spectrum to match PB fluxes {{{
# Usage: bkg_rescale($src_spec, $bkg_spec)
sub bkg_rescale {
    my $argc = @_;
    if ($argc != 2) {
        print "ERROR: sub bkg_rescale() requires 2 arguments!\n";
        exit 22;
    }
    my $src_spec = $_[0];
    my $bkg_spec = $_[1];

    # PB counts of source & background spectrum
    my $src_pb_cnt = `punlearn dmstat && dmstat infile="$src_spec\[channel=$pb_chn_lo:$pb_chn_hi\]\[cols COUNTS\]" >/dev/null 2>&1 && pget dmstat out_sum`;
    chomp($src_pb_cnt);
    # PB counts of background spectrum
    my $bkg_pb_cnt = `punlearn dmstat && dmstat infile="$bkg_spec\[channel=$pb_chn_lo:$pb_chn_hi\]\[cols COUNTS\]" >/dev/null 2>&1 && pget dmstat out_sum`;
    chomp($bkg_pb_cnt);
    #print "src_pb_cnt: $src_pb_cnt\n";
    #print "bkg_pb_cnt: $bkg_pb_cnt\n";
    #exit 233;

    # BACKSCAL & EXPOSURE of source spectrum
    my $src_backscal = `punlearn dmkeypar && dmkeypar $src_spec BACKSCAL echo=yes`;
    chomp($src_backscal);
    my $src_exposure = `punlearn dmkeypar && dmkeypar $src_spec EXPOSURE echo=yes`;
    chomp($src_exposure);
    # BACKSCAL & EXPOSURE of background spectrum
    my $bkg_backscal = `punlearn dmkeypar && dmkeypar $bkg_spec BACKSCAL echo=yes`;
    chomp($bkg_backscal);
    my $bkg_exposure = `punlearn dmkeypar && dmkeypar $bkg_spec EXPOSURE echo=yes`;
    chomp($bkg_exposure);
    #print "src_backscal: $src_backscal\n";
    #print "src_exposure: $src_exposure\n";
    #print "bkg_backscal: $bkg_backscal\n";
    #print "bkg_exposure: $bkg_exposure\n";

    if ($src_pb_cnt <= $pb_cnt_threshold) {
        # Source spectrum particle background counts too small.
        print "WARNING: too small counts of source spectrum within 9.5-12 keV: $src_pb_cnt!\n";
        print "Skipped BACKSCAL adjustment of background spectrum!\n";
    } else {
        # Adjust BACKSCALE of background spectrum to match PB flux:
        # src_pb_cnt / src_exposure / src_backscal = bkg_pb_cnt / bkg_exposure / bkg_backscal
        my $bkg_backscal_new = $src_backscal * $src_exposure * $bkg_pb_cnt
                / ($src_pb_cnt * $bkg_exposure);
        my $cmd_str = "punlearn dmhedit && "
            . "dmhedit infile=\"$bkg_spec\" filelist=none operation=add "
            . "key=BACKSCAL value=$bkg_backscal_new "
            . "comment='old_value: $bkg_backscal'";
        system($cmd_str);
        print "Updated BACKSCAL: $bkg_backscal -> $bkg_backscal_new\n";
    }
}
# rescale }}}

# Extract spectrum from fits {{{
# Usage: extract_spectrum($input_fits, $output_spec, $region_file)
sub extract_spectrum {
    my $argc = @_;
    if ($argc != 3) {
        print "ERROR: sub extract_spec() requires 3 arguments!\n";
        exit 31;
    }
    my $infile  = $_[0];
    my $outfile = $_[1];
    my $regfile = $_[2];
    my $cmd_str = "punlearn dmextract && "
        . "dmextract infile=\"$infile\[sky=region\($regfile\)\]\[bin pi\]\" "
        . "outfile=\"$outfile\" wmap=\"\[bin det=8\]\" clobber=yes";
    system($cmd_str);
}
# extract spectrum }}}

# group spectrum with 'dmgroup' {{{
# Usage: group_spectrum($in_spec, $grp_spec, $grouptype, $grouptypeval, $binspec)
sub group_spectrum {
    my $argc = @_;
    if ($argc != 5) {
        print "ERROR: sub group_spec() requires 5 arguments!\n";
        exit 41;
    }
    my $infile  = $_[0];
    my $outfile = $_[1];
    my $grp_type = $_[2];
    my $grp_type_val = $_[3];
    my $bin_spec = $_[4];

    my $cmd_str = "punlearn dmgroup && "
        . "dmgroup infile=\"$infile\" outfile=\"$outfile\" "
        . "grouptype=\"$grp_type\" grouptypeval=$grp_type_val "
        . "binspec=\"$bin_spec\" xcolumn=CHANNEL ycolumn=COUNTS clobber=yes";
    #print $cmd_str;
    system($cmd_str);
}
# group spectrum }}}

# Parse the fitted result line {{{
# Usage: parse_result($result_line, @para_name_list, %para_num)
sub parse_result ($\@\%) {
    my $argc = @_;
    if ($argc != 3) {
        print "ERROR: sub parse_result() requires 3 argument!\n";
        exit 51;
    }
    my $line=$_[0];
    my @para_name_list = @{$_[1]};
    my %para_num= %{$_[2]};

    my @parsed_results = ();

    foreach my $para_name (@para_name_list) {
        # Number of values for this parameter
        my $num = $para_num{$para_name};
        if ($line =~ /^$para_name:/) {
            my @items = split(/\s+/, $line);
            foreach my $i (0 .. $num) {
                # items[0] is the '$para_name:'
                push(@parsed_results, $items[$i]);
            }
            if ($para_name =~ /red.*chi.*sq/i) {
                my $current_rechisq = $items[1];
                if ($current_rechisq > $rechisq_threshold) {
                    print "WARNING: bad quality of spectral fittings!\n";
                    $bad_fit_num ++;
                }
            }
        }
    }

    return @parsed_results;
}
# parse result }}}
## subroutines }}}

## Generate the fit script for XSPEC {{{
if (! -e $config_fn) {
    print "ERROR: config file '$config_fn' does not exist!\n";
    exit 14;
}
open(my $FIT_SCRIPT, ">", $fit_script_fn)
    or die "ERROR: Could not open file '$fit_script_fn', $!";

# Part of load data to XSPEC
foreach my $i (0 .. $#evt_list) {
    my $ii = $i + 1;
    print $FIT_SCRIPT "data ${ii}:${ii} $fit_grp_spec_list[$i]\n";
    print $FIT_SCRIPT "response 1:${ii} $rmf_list[$i]\n";
    print $FIT_SCRIPT "arf 1:${ii} $arf_list[$i]\n";
    print $FIT_SCRIPT "backgrnd ${ii} $fit_bkg_spec_list[$i]\n";
}

# Automatically answer XSPEC query with 'yes'
print $FIT_SCRIPT "query yes\n";
print $FIT_SCRIPT "set ff \[ open $fit_result_fn w \]\n";

open(my $CONFIG, "<", $config_fn)
    or die "ERROR: Counld not open file '$config_fn', $!";
my @config_lines = <$CONFIG>;
close($CONFIG);

# Record the name of specified parameters in the config file.
my @para_name_list = ();
# Record the number of output values of above parameters. (use hash)
my %para_num;

# Parse the fitting config file {{{
foreach my $line (@config_lines) {
    chomp($line);
    if ($line =~ /^\s*$/) {
        # Blank line
        print "Skipped blank line.\n";
        next;
    } elsif ($line =~ /^\/\//) {
        # Comment line
        print "Skipped comment line.\n";
        next;
    } elsif ($line =~ /^\#PARA/) {
        # Define fitting parameter, together with settings error calculation.
        # Config: '#PARA para_name para_index [ ERROR conf_level ]'
        # Output:
        #     'para_name: value sigma'
        #     'para_name_err: error_lower error_upper'
        my @para = split(/\s+/, $line);
        my $para_name = $para[1];
        my $para_idx = $para[2];
        push(@para_name_list, $para_name);
        $para_num{$para_name} = 2;
        print $FIT_SCRIPT "tclout para $para_idx\n";
        print $FIT_SCRIPT "scan \$xspec_tclout \"%f\" value\n";
        print $FIT_SCRIPT "tclout sigma $para_idx\n";
        print $FIT_SCRIPT "scan \$xspec_tclout \"%f\" sigma\n";
        print $FIT_SCRIPT "puts \$ff \"$para_name: \$value \$sigma\"\n";
        # Check whether to calculate the errors
        if ($line =~ /ERR(OR|)\s+\d*\.\d*\s*$/) {
            # Calculate errors with XSPEC 'error' command
            my $conf_level = $para[4];
            my $para_err_name = "${para_name}_err";
            push(@para_name_list, $para_err_name);
            $para_num{$para_err_name} = 2;
            print $FIT_SCRIPT "fit\n";
            print $FIT_SCRIPT "error $conf_level $para_idx\n";
            print $FIT_SCRIPT "tclout error $para_idx\n";
            print $FIT_SCRIPT "scan \$xspec_tclout \"%f %f\" err_lower err_upper\n";
            print $FIT_SCRIPT "puts \$ff \"${para_name}_err: \$err_lower \$err_upper\"\n";
        }
    } elsif ($line =~ /^\#STAT/) {
        # Whether output the statistic information
        # Config: '#STAT'
        # Output:
        #     'stat_value statistic'
        #     'dof_value dof'
        #     'rechisq_value Reduced_chisq'
        my $stat_name = "stat";
        push(@para_name_list, $stat_name);
        $para_num{$stat_name} = 1;
        my $dof_name  = "dof";
        push(@para_name_list, $dof_name);
        $para_num{$dof_name} = 1;
        # Reduced chi-squared
        my $rechisq_name = "reduced_chisq";
        push(@para_name_list, $rechisq_name);
        $para_num{$rechisq_name} = 1;
        print $FIT_SCRIPT "tclout $stat_name\n";
        print $FIT_SCRIPT "scan \$xspec_tclout \"%f\" stat_val\n";
        print $FIT_SCRIPT "puts \$ff \"$stat_name: \$stat_val\"\n";
        print $FIT_SCRIPT "tclout $dof_name\n";
        print $FIT_SCRIPT "scan \$xspec_tclout \"%d\" dof_val\n";
        print $FIT_SCRIPT "puts \$ff \"$dof_name: \$dof_val\"\n";
        print $FIT_SCRIPT "set rechisq_cmd \{ format \"%.5f\" \[ expr \{ \$stat_val / \$dof_val \} \] \}\n";
        print $FIT_SCRIPT "set rechisq \[ eval \$rechisq_cmd \]\n";
        print $FIT_SCRIPT "puts \$ff \"$rechisq_name: \$rechisq\"\n";
    } elsif ($line =~ /^\#FLUX/) {
        # Whether output flux information
        # Config: '#FLUX start_keV end_keV'
        my @para = split(/\s+/, $line);
        my $energy_low = $para[1];
        my $energy_high = $para[2];
        my $para_name = "flux";
        push(@para_name_list, $para_name);
        $para_num{$para_name} = 1;
        print $FIT_SCRIPT "flux $energy_low $energy_high\n";
        print $FIT_SCRIPT "tclout $para_name\n";
        print $FIT_SCRIPT "scan \$xspec_tclout \"%f\" flux_val\n";
        print $FIT_SCRIPT "puts \$ff \"$para_name: \$flux_val\"\n";
    } elsif ($line =~ /^\#DMGROUP/) {
        # dmgroup parameters
        my @para = split(/\s+/, $line);
        $grouptype = $para[1];
        if ($grouptype eq "NUM_CTS") {
            $grouptypeval = $para[2];
            print "dmgroup: $grouptype, $grouptypeval\n";
        } elsif ($grouptype eq "BIN") {
            $binspec = $para[2];
            print "dmgroup: $grouptype, $binspec\n";
        } else {
            print "ERROR: unknown grouptype '$grouptype'!\n";
            print "Only 'NUM_CTS' and 'BIN' supported for dmgroup.\n";
            exit 16;
        }
    } elsif ($line =~ /^\#NORESCALE/) {
        $rescale = 0;
    } elsif ($line =~ /^\#/) {
        # Other unsupported lines starting with '#'
        print "WARNING: unsupported config line:\n    '$line'\n";
        next;
    } else {
        # XSPEC commands/lines
        print $FIT_SCRIPT "$line\n";
    }
}
# Parse config file }}}

print $FIT_SCRIPT "close \$ff\n";
print $FIT_SCRIPT "tclexit\n";
close($FIT_SCRIPT);
#exit 233;
## Generate XSPEC script }}}

## Perform spectral fittings {{{
open(my $REGION_LIST, "<", $region_list_fn)
    or die "ERROR: Could not open file '$region_list_fn', $!";
my @region_lines = <$REGION_LIST>;
close($REGION_LIST);

open(my $OUTFILE, ">", $output_fn)
    or die "ERROR: Could not open file '$output_fn', $!";

my $reg_total = @region_lines;
my $n = 0;

# Loop over the region list to perform spectral fitting for each region
foreach my $line (@region_lines) {
    chomp($line);
    $n++;
    print "### $n of $reg_total ###\n";

    # Skip null/none regions
    if ($line =~ /^\#(null|none)/i) {
        print $OUTFILE "$n ";
        foreach my $para_name (@para_name_list) {
            # Number of values for this parameter
            my $num = $para_num{$para_name};
            print $OUTFILE "${para_name}: ";
            foreach my $i (1 .. $num) {
                print $OUTFILE "-1 ";
            }
        }
        print $OUTFILE "\n";
        print "Skipped a null/none region.\n";
        next;
    }

    # Save the current region to a temporary region file
    unlink($fit_reg_fn);
    if (-f $line) {
        # Each line is a region file
        copy($line, $fit_reg_fn)
            or die "ERROR: Could not copy region file '$line', $!";
    } else {
        # Each line is a region string
        open(my $FIT_REG, ">", $fit_reg_fn)
            or die "ERROR: Could not open file '$fit_reg_fn', $!";
        my @cur_regs = split(/\s+/, $line);
        foreach my $reg (@cur_regs) {
            print $FIT_REG "$reg\n"
        }
        close($FIT_REG);
    }

    # Prepare source and background spectra {{{
    print "Prepare source and background  spectra \.\.\.\n";
    foreach my $i (0 .. $#evt_list) {
        # Prepare source spectrum
        extract_spectrum($evt_list[$i], $fit_spec_list[$i], $fit_reg_fn);
        group_spectrum($fit_spec_list[$i], $fit_grp_spec_list[$i],
            $grouptype, $grouptypeval, $binspec);
        # Prepare background spectrum
        if ($bkg_type[$i] == NO_BKG) {
            # Dose not use a background
            $fit_bkg_spec_list[$i] = "none";
        } elsif (($bkg_type[$i] == BLANKSKY) || ($bkg_type[$i] == STOW_BKG)) {
            extract_spectrum($bkg_list[$i], $fit_bkg_spec_list[$i], $fit_reg_fn);
            if ($rescale) {
                bkg_rescale($fit_spec_list[$i], $fit_bkg_spec_list[$i]);
            }
        } elsif ($bkg_type[$i] == LBKG_REG) {
            extract_spectrum($evt_list[$i], $fit_bkg_spec_list[$i], $bkg_list[$i]);
            if ($rescale) {
                bkg_rescale($fit_spec_list[$i], $fit_bkg_spec_list[$i]);
            }
        } elsif ($bkg_type[$i] == BKG_SPEC) {
            # Directly use given background spectrum
            system("cp -v $bkg_list[$i] $fit_bkg_spec_list[$i]");
            if ($rescale) {
                bkg_rescale($fit_spec_list[$i], $fit_bkg_spec_list[$i]);
            }
        } else {
            print "ERROR: unsupported background type!\n";
            exit 15;
        }
    }
    # prepare spectra }}}

    # Perform the spectral fitting with XSPEC
    print "Fitting \.\.\.\n";
    unlink($fit_result_fn);
    system("xspec - $fit_script_fn >/dev/null") == 0
        or die "system: xspec failed: $?";

    # Parse fitted results and output
    open(my $FIT_RESULT, "<", $fit_result_fn)
        or die "ERROR: Could not open file '$fit_result_fn', $!";
    my @fitted_results = <$FIT_RESULT>;
    close($FIT_RESULT);
    # Array to store parsed results of current fit
    my @current_results = ($n);
    foreach my $line (@fitted_results) {
        chomp($line);
        push(@current_results, parse_result($line, @para_name_list, %para_num));
    }
    print join(" ", @current_results) . "\n";
    print $OUTFILE join(" ", @current_results) . "\n";
} # end of fitting loop

close($OUTFILE);
## spectra fittings }}}

if ($bad_fit_num > 0) {
    my $bad_fit_percent = $bad_fit_num / $reg_total;
    print "\n************************************************\n";
    print "WARNING: Reduced chi-squared of $bad_fit_num ($bad_fit_percent) regions > $rechisq_threshold!!\n";
}

exit 0;
