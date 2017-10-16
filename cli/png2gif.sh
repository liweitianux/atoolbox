#!/bin/sh

#convert ???.png -background white -alpha remove -resize 50% -layers optimize -delay 5 -loop 0 simu.gif

[ ! -d gifs ] && mkdir gifs

for f in ???.png; do
    convert $f -trim -resize 50% gifs/${f%.png}.gif
done
gifsicle --delay=5 --loop --colors=256 --optimize=3 gifs/???.gif > simu.gif

