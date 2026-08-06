# Experimental data references

This directory preserves source material and provenance for experimental
aerodynamic-coefficient data. Nothing here is read by the training workflow;
the machine-readable inputs remain in `../case-dat/`.

## Sheldahl and Klimas

`sheldahl-klimas-sand80-2114.pdf` is:

R. E. Sheldahl and P. C. Klimas, *Aerodynamic Characteristics of Seven
Symmetrical Airfoil Sections Through 180-Degree Angle of Attack for Use in
Aerodynamic Analysis of Vertical Axis Wind Turbines*, Sandia National
Laboratories, SAND80-2114, 1981. DOI: <https://doi.org/10.2172/6548367>.

Official record: <https://www.osti.gov/biblio/6548367>

The report supports the processed `sheld-n0012.csv` and `sheld-n0021.csv`
datasets. It combines wind-tunnel measurements with calculated/extrapolated
values, so the tables should not all be described as direct experimental
measurements.

## NACA 0012 data

The files under `nasa/` preserve small, machine-readable datasets:

- `n0012-CLCD_Ladson_expdata-dat.txt` contains tripped-transition NACA 0012
  lift and drag measurements at Reynolds number 6 million and Mach 0.15 from
  C. L. Ladson, NASA-TM-4074 (1988). Official report:
  <https://ntrs.nasa.gov/citations/19880019495>.
- `n0012-abbottdata-cl-dat.txt` and `n0012-abbottdata-cd-dat.txt` contain
  approximately digitised NACA 0012 plots at Reynolds number 6 million from
  Abbott and von Doenhoff, *Theory of Wing Sections*. The approximation warning
  in the files should be retained when these values are used.

These files are retained for validation and provenance but are not currently
loaded by `main.py`.

## Unverified CFD tables

The images under `unverified-cfd-tables/` are Greek-language tables comparing
NACA 0012 CFD/turbulence-model outputs with experimental values, including
air-water cases at Reynolds number 3 million. Their original publication and
licensing information were not present in the old project. Treat them as
unverified reference material and do not use them as authoritative training
data without identifying the source.
