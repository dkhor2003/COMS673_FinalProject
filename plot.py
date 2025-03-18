import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os

parser=ArgumentParser()
parser.add_argument('-i','--in_csv', required=True)
parser.add_argument('-o','--out_png', required=True)

args=parser.parse_args()
in_csv=args.in_csv
out_png=args.out_png

df=pd.read_csv(in_csv)

fallover=df['fallover']
df.drop(columns=['fallover'],inplace=True)
fall_start=fallover[fallover.diff()==1].index

df=(df-df.mean())/df.std()

fig, ax = plt.subplots()

df.plot(ax=ax,legend=False)
if len(fall_start)>0:
    ax.axvspan(fall_start[0],len(df), color='red', alpha=0.5)

plt.xlabel('step')
plt.ylabel('normalized attributes')

plt.savefig(out_png)