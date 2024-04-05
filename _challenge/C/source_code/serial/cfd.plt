set terminal pngcairo
set output 'cfd_output.png'
set size square
set key off
unset xtics
unset ytics
set xrange [-63:2112]
set yrange [-63:2112]
plot "colourmap.dat" w rgbimage, "velocity.dat" u 1:2:(64*0.75*$3/sqrt($3**2+$4**2)):(64*0.75*$4/sqrt($3**2+$4**2)) with vectors  lc rgb "#7F7F7F"