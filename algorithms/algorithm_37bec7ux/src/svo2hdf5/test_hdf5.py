import h5py

file_name = "output.hdf5"
hf = h5py.File(file_name, "r")

print(hf['depth_maps'][0])
print(hf['images'])
