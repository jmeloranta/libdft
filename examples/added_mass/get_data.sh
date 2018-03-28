## v=0
#echo -ne 0.000000'\t'
#tail -4 res0 | awk '/ener/ {print $5}'

# This returns a list with kinetic energy per particle , total energy
for r in `ls ebub_mass_*/res_* | sort -V`
do
	awk '/background kinetic/ {ek=$6} /helium natoms/ {ek/=$6} /impurity kinetic/ {en=$6} /helium energy/ {en+=$6} END {print ek,en}' $r
done
