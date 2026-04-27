from cddiagram import draw_cd_diagram
import pandas as pd
import scikit_posthocs as sp
import matplotlib.pyplot as plt

values = pd.read_csv("examples/test_data.csv", index_col=0, header=0)


draw_cd_diagram(values, labels=values.columns.to_list(), out_file="out.svg", title="Model comparison")