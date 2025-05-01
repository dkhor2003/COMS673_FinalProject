for in_csv in data/g1_traj_*/*.csv; do
    out_png="${in_csv%.csv}.png"
    echo "Plotting $in_csv to $out_png"
    python plot.py --in_csv $in_csv --out_png $out_png
done