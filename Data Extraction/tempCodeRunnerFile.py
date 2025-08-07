
curr_idx=2139
for idx in range(len(series)):
    filename=f"Spam_{idx+curr_idx:05}.txt"
    output_path=os.path.join("./Spam",filename)
    with open(output_path,"w") as f:
        f.write(series.iloc[idx][:-8])