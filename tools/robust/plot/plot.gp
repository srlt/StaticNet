set terminal svg size 600,400 dynamic enhanced butt solid background "#ffffff"

if (!exists("filename")) filename='default.dat'
set output filename.".svg"

set xlabel "Byzantine neurons"
set ylabel "Error per output"
set xrange [1:]
set xtics  add ("1" 1)

set style fill transparent solid 0.1 border
set grid noxtics nomxtics noytics nomytics front

plot filename with filledcurve y1 lc "#c00000" title "Error possible domain"
