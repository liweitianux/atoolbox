#!/bin/sh
#
# MP3 id3tag functions
#
# Copyright (C) 2005 Robert, Zhang Le
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


# INTRO:
#	 This file contains several functions related to mp3 file's operations.
#	 I wrote them in order to ease the process of transforming the encoding 
#	 of mp3 file's id3tag from gbk to utf8. YOU NEED id3v2 to make these 
#	 functions work.
#    !!! USE THESE FUNCTIONS IN UTF-8 LOCALE !!!

# TODO: 
#	1. make all the functions work both on a directory and a file
#	2. add testing feature to all functions
#      * rename_delete_ending_artist()
#      * check_artist()
#	3. set_artist() maybe need to be improved
#   4. if functions are called without params, echo "usage"

# USAGE:
#	1. source this file before use


# get_title_1() can be used to get v1 title from an mp3 file's id3tag
# get_title_1_convernted() convert its encoding from gbk to utf8

get_title_1(){
	id3v2 -l "$1" | grep ^Title \
	| sed -e 's/^Title  : //g' -e 's/[[:space:]]*Artist: [[:print:]]*//g'
}
get_title_1_converted(){
	id3v2 -l "$1" | grep ^Title \
	| sed -e 's/^Title  : //g' -e 's/[[:space:]]*Artist: [[:print:]]*//g' \
	| iconv -f gbk -t utf8 
}

# ATTENTION: if title encoding is gbk, the following function does not
#            function ideally, because "." can't match invisible/unmappd char?
get_artist_1(){
	id3v2 -l "$1" | grep ^Title \
	| sed -e  's/^Title.*Artist: //g' 
}

get_title_2(){
	id3v2 -l "$1" | grep ^TIT2 | tail -n 1 | sed -e 's/^.*: //g'
}
get_title_2_converted(){
	id3v2 -l "$1" | grep ^TIT2 | tail -n 1 | sed -e 's/^.*: //g' \
	| iconv -f gbk -t utf8 

}
get_artist_2(){
	id3v2 -l "$1" | grep ^TPE1 | tail -n 1 | sed -e 's/^.*: //g'
}


# if only one param, it must be a dirname, and at the same time an artist's name
# if two param, the first artist name, the second dirname
set_artist(){ 
	local artist 

	if [[ $# -eq 0 ]]; then
		echo "set_artist artist [dir]"
		return;
	else
		artist="$1";
	fi

	if [[ $# -eq 2 ]]; then
		if [[ -d $2 ]]; then
			shift;
		else
			echo "the 2nd param must be a dir"
			return;
		fi
	elif [[ $# -eq 1 ]]; then
		if [[ ! -d $1 ]]; then
			echo "the sole param must be also a dir"
			return;
		fi
	fi

	find "$1" -name "*.mp3" -print0 | xargs -0 id3v2 -a "$artist" ;

}

id3_convert_set_title(){
	local target
	local title
	local testing 
	local tmp
	local i

	if [[ $# -eq 0 ]]; then
		echo "needs parameters"
		return;
	fi

	if [[ $1 == "notest" ]]; then
		testing=0;
		shift;
	else
		testing=1;
	fi


	while [[ $# -gt 0 ]]; do
		if [[ -d "$1" ]]; then
			for i in "$1"/*.mp3; 
			do 
				title=`get_title_1_converted "$i"`

				# if the last command, ie. iconv fails
				# then probably the file's id3tag has already been converted
				if [[ $? -ne 0 ]]; then
					continue;
				fi

				tmp=`echo -n "$title" | sed -e 's/Track//g' -e 's/[[:space:][:digit:]]*//g'`

				if [[ -z $tmp ]]; then
					title=`get_title_2_converted "$i"`

					if [[ $? -ne 0 ]]; then
						continue;
					fi

					tmp=`echo -n "$title" | sed -e 's/Track//g' -e 's/[[:space:][:digit:]]*//g'`

					if [[ -z $tmp ]]; then
						echo "title is "$tmp" nothing need to be done"
						continue;
					fi
				fi

				if [[ -n $title ]]; then
					if [[ $testing -eq 1 ]]; then
						echo $title;
					else
						echo -n $title;
						id3v2 -1 -t "$title" "$i"; 
						id3v2 -2 -t "$title" "$i"; 
						echo " done";
					fi
				fi
			done
		elif [[ -f "$1" ]]; then
			title=`get_title_1_converted "$1"`
			if [[ $? -ne 0 ]]; then
				shift;
				continue;
			fi

			tmp=`echo -n "$title" | sed -e 's/Track//g' -e 's/[[:space:][:digit:]]*//g'`

			if [[ -z $tmp ]]; then
				title=`get_title_2_converted "$1"`

				if [[ $? -ne 0 ]]; then
					continue;
				fi

				tmp=`echo -n "$title" | sed -e 's/Track//g' -e 's/[[:space:][:digit:]]*//g'`

				if [[ -z $tmp ]]; then
					echo "title is "$tmp" nothing need to be done"
					shift;
					continue;
				fi
			fi

			if [[ -n $title ]]; then
				if [[ $testing -eq 1 ]]; then
					echo $title;
				else
					echo -n $title;
					id3v2 -1 -t "$title" "$1"; 
					id3v2 -2 -t "$title" "$1"; 
					echo " done";
				fi
			fi
		fi
		shift;
	done

	if [[ $testing -eq 1 ]]; then
		echo "testing finished, to convert use \"notest\" as the fist param"
	fi
}

# make sure file name ends with ".mp3"
id3_set_title_as_filename(){
	local title 
	local testing
	local i

	if [[ $# -eq 0 ]]; then
		echo "needs parameters";
		return;
	fi

	if [[ $1 == "notest" ]]; then
		testing=0;
		shift;
	else
		testing=1;
	fi

	while [[ $# -gt 0 ]]; do
		if [[ -d "$1" ]]; then
			for i in "$1"/*.mp3; do 
				title=`basename "$i" | sed -e 's/\.mp3//g'`	
				if [[ $testing -eq 1 ]]; then
					echo $title;
				else
					echo -n $title;
					id3v2 -1  -t "${title}" "$i";
					id3v2 -2  -t "${title}" "$i";
					echo " done";
				fi
			done 
		elif [[ -f "$1" ]]; then
			title=`basename "$1" | sed -e 's/\.mp3//g'`
			if [[ $testing -eq 1 ]]; then
				echo $title;
			else
				echo -n $title;
				id3v2 -1 -t "$title" "$1"; 
				id3v2 -2 -t "$title" "$1"; 
				echo " done";
			fi
		fi
		shift;
	done

	if [[ $testing -eq 1 ]]; then
		echo "testing finished, to convert use \"notest\" as the fist param"
	fi
}

id3_rename_MP3_to_mp3(){
	local target
	local i

	if [[ $# -eq 0 ]]; then
		echo "needs parameters";
		return;
	fi

	while [[ $# -gt 0 ]]; do
		if [[ -d "$1" ]]; then
			for i in "$1"/*.MP3; do
				target="$1"/`basename "$i" | sed -e 's/MP3$/mp3/g'`
				if [[ "$i" != "$target" ]]; then
					mv "$i" "$target"; 
					echo "$i -> $target: done"
				fi
			done
		elif [[ -f "$1" ]]; then
			target=`dirname "$1"`/`basename "$1" | sed -e 's/MP3$/mp3/g'`
			if [[ "$1" != "$target" ]]; then
				mv "$1" "$target"; 
				echo "$1 -> $target: done"
			fi
		fi
		shift;
	done
}

# use this function to remove leading garbages in the file name
# Sometimes, the file name in the form of "artist -(1). title.mp3"
# we just need title.mp3
# the delim is the adjacent char before "title"

id3_rename_delete_leading_garbage(){
	local target
	local testing
	local orig="^0[0-9]\."
	#local orig="^[[:alnum:]]*[-.()[:digit:][:space:]]*"
	local delim="[[:space:]]"
	# ATTENTION: if delimeter is  ".",  delim should be  "\."
	#            when passing on command line, it should be "\\\."
	local new=""
	local i

	if [[ $# -eq 0 ]]; then
		echo "needs parameters";
		return;
	fi

	if [[ $1 == "notest" ]]; then
		testing=0;
		shift;
	else
		testing=1;
	fi

	if [[ "${1/=*/}" == "delim" ]]; then
		delim="${1/*=/}";
		echo $delim;
		shift;
	fi

	while [[ $# -gt 0 ]]; do
		if [[ -d "$1" ]]; then
			echo $1
			for i in "$1"/*.mp3; do
				target="$1"/`basename "$i" | sed -e "s/$orig$delim/$new/g"`	
				if [[ "$i" != "$target" ]]; then
					echo -n "$target";
					if [[ $testing -eq 0 ]]; then
						mv "$i" "$target"; 
					fi
					echo " done"
				else
					echo "$target";
				fi
			done
		elif [[ -f "$1" ]]; then
			target=`dirname "$1"`/`basename "$1" | sed -e "s/$orig$delim/$new/g"`	
			if [[ "$1" != "$target" ]]; then
				echo -n "$target";
				if [[ $testing -eq 0 ]]; then
					mv "$1" "$target"; 
				fi
				echo " done";
			else
				echo $target;
			fi
		fi
		shift;
	done

	if [[ $testing -eq 1 ]]; then
		echo "testing finished, to convert use \"notest\" as the fist param"
	fi
}

# use this function after conversion
id3_rename_as_title(){
	local target
	local title
	local testing
	local i

	if [[ $# -eq 0 ]]; then
		echo "needs parameters";
		return;
	fi

	if [[ $1 == "notest" ]]; then
		testing=0;
		shift;
	else
		testing=1;
	fi

	while [[ $# -gt 0 ]]; do
		if [[ -d "$1" ]]; then
			for i in "$1"/*.mp3; do
				title=`get_title_1 "$(basename $i)"`

				# if the last command, ie. iconv fails
				# then probably the file's id3tag has already been converted
				if [[ $? -ne 0 ]]; then
					continue;
				fi

				title=`echo -n "$title" | sed -e 's/[[:space:][:punct:]]*//g'`

				if [[ -z $title ]]; then
					title=`get_title_2_converted "$(basename $i)"`
					if [[ $? -ne 0 ]]; then
						continue;
					fi
				fi

				target="$1"/`get_title_2 "$i"`.mp3

				if [[  -a "$target" ]]; then
					echo "$target exists, skiping"
					continue;
				fi

				if [[ "$i" != "$target" ]] ; then
					if [[ $testing -eq 0 ]]; then
						mv "$i" "$target"; 
					fi
					echo "$i -> $target: done";
				fi
			done
		elif [[ -f "$1" ]]; then
			target=`dirname "$1"`/`get_title_2 "$1"`.mp3

			if [[  -a "$target" ]]; then
				echo "$target exists, skiping"
				shift;
				continue;
			fi

			if [[ "$1" != "$target" ]]; then
				if [[ $testing -eq 0 ]]; then
					mv "$1" "$target"; 
				fi
				echo "$1 -> $target: done";
			fi
		fi
		shift;
	done

	if [[ $testing -eq 1 ]]; then
		echo "testing finished, to convert use \"notest\" as the fist param"
	fi
}

id3_check_title(){
	local target
	local i
	local title1
	local title2

	if [[ $# -eq 0 ]]; then
		echo "needs parameters";
		return;
	fi

	while [[ $# -gt 0 ]]; do
		if [[ -d "$1" ]]; then
			echo $1
			for i in "$1"/*.mp3
			do
				title1=`get_title_1 "$i"`;
				title2=`get_title_2 "$i"`;
				printf "%s   \tv1: %s   \tv2: %s\n" "$i" "$title1" "$title2";
			done
		elif [[ -f "$1" ]]; then
			title1=`get_title_1 "$1"`;
			title2=`get_title_2 "$1"`;
			printf "%s   \tv1: %s   \tv2: %s\n" "$1" "$title1" "$title2";

		fi
		shift;
	done
}

id3_check_title_converted(){
	local target
	local i
	local title1
	local title2

	if [[ $# -eq 0 ]]; then
		echo "needs parameters";
		return;
	fi

	while [[ $# -gt 0 ]]; do
		if [[ -d "$1" ]]; then
			echo $1
			for i in "$1"/*.mp3
			do
				title1=`get_title_1_converted "$i"`;
				title2=`get_title_2_converted "$i"`;
				printf "%s   \tv1: %s   \tv2: %s\n" "$i" "$title1" "$title2";
			done
		elif [[ -f "$1" ]]; then
			title1=`get_title_1_converted "$1"`;
			title2=`get_title_2_converted "$1"`;
			printf "%s   \tv1: %s   \tv2: %s\n" "$1" "$title1" "$title2";

		fi
		shift;
	done
}

id3_check_artist(){
	for i in *.mp3;
	do
		artist1=`get_artist_1 "$i"`;
		artist2=`get_artist_2 "$i"`;
		printf "%s   \tv1: %s   \tv2: %s\n" "$i" "$artist1" "$artist2";
	done
}
# if filename ends with "_artist", use this one to remove the suffix
id3_rename_delete_ending_artist(){

	if [[ $# -eq 0 ]]; then
		echo "needs parameters";
		return;
	fi

	for i in "$1"/*.mp3; do
		mv "$i" "${i/_*./.}";
	done
}

