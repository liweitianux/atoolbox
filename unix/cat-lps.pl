#!/usr/bin/env perl
#
# Cat file to terminal at particular speed of lines per second
# https://superuser.com/a/526249
#
# Usage: cat-lps.pl [lps] [file]...
#

use warnings;
use strict;
use Time::HiRes qw|time|;

my $start=time;
my $lps=300;

$lps=shift @ARGV if @ARGV && $ARGV[0]=~/^(\d+)$/;
my $sleepPerLine=1/$lps;

print &&
    select undef,undef,undef,($start + $sleepPerLine*$. - Time::HiRes::time)
    while <>
