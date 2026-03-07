import wfdb

# Download one AFDB record
wfdb.dl_database('afdb', dl_dir='Data', records=['07879', '05121'])

print("Download complete!")