# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from notebook_utils import *
import numpy as np
import pickle
import zarr
from datetime import datetime
from dtaidistance import dtw, dtw_ndim, dtw_visualisation


# %%
# CONSTANTS
AT1='austria/33UVP/2017'
DK1='denmark/32VNH/2017'
FR1='france/30TXT/2017'
FR2='france/31TCJ/2017'

region_aliases = {
    AT1: 'AT1',
    DK1: 'DK1',
    FR1: 'FR1',
    FR2: 'FR2'
}

pos_type_aliases = {
    'fourier': 'tpe_fourier',
    'rnn': 'tpe_recurrent'
}

classes = sorted(['corn', 'horsebeans', 'meadow', 'spring_barley', 'unknown',
                   'winter_barley', 'winter_rapeseed', 'winter_triticale', 'winter_wheat'])


# %% [markdown]
# Define experiments parameters here. (Source region(s), target and type of positional encoding)

# %%
# EXPERIMENT PARAMETERS
SOURCE = [AT1] # Source tile(s) parameter must be enclosed in a list as per code convention
TARGET = FR2
POS_TYPE = 'rnn'

# Select crop_id for which to generate illustrations
crop_id = classes.index('winter_barley')

# %%
SOURCE_STR = '+'.join([region_aliases[s] for s in SOURCE])
EXP = f"pseltae_{SOURCE_STR}_{pos_type_aliases[POS_TYPE]}"

config = create_config(EXP, SOURCE, TARGET, POS_TYPE, notebook=False)
EXP = config.experiment_name
SOURCE = config.source
TARGET = config.target[0]

print("Loading data...")
source_loader = create_train_loader(config.source, config)
target_test_loader = create_test_loader(config.target, config, random_sample_time_steps=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
output_dir = os.path.join("../outputs", config.experiment_name)
fold_dir = os.path.join(output_dir, "fold_0")
model = load_model(config, fold_dir, device)

# %%
source_data_filename = f"{config.output_dir}/{classes[crop_id]}_source_data.pkl"

if os.path.exists(source_data_filename):
    with open(source_data_filename, 'rb') as f:
        source_data = pickle.load(f)
else:
    source_data = get_crop_data(crop_id, source_loader, model, device)
    with open(source_data_filename, 'wb') as f:
        pickle.dump(source_data, f)

# %%
target_data_filename = f"{config.output_dir}/{region_aliases[TARGET]}_{classes[crop_id]}_target_data.pkl"

if os.path.exists(target_data_filename):
    with open(target_data_filename, 'rb') as f:
        target_data = pickle.load(f)
else:
    target_data = get_crop_data(crop_id, target_test_loader, model, device)
    with open(target_data_filename, 'wb') as f:
        pickle.dump(target_data, f)

# %%
# Once the data is loaded, we compute a DTW distance matrix to find the most dissimilar PE
dtw_matrix_filename = f"{config.output_dir}/{region_aliases[TARGET]}_{classes[crop_id]}_dtw_distance_matrix.pkl"
if not os.path.exists(dtw_matrix_filename):

    dtw_distance_matrix = np.zeros((len(source_data), len(target_data)))
    for i in tqdm(range(len(source_data))):
        for j in range(len(target_data)):
            dtw_distance_matrix[i][j] = dtw_ndim.distance_fast(source_data[i]['pe'], target_data[j]['pe'])

    with open(dtw_matrix_filename, "wb") as f:
        pickle.dump(dtw_distance_matrix, f)
else:
    with open(dtw_matrix_filename, "rb") as f:
        dtw_distance_matrix = pickle.load(f)

# %%
# Get the indices of the samples with the highest DTW distance
argmax = dtw_distance_matrix.argmax()
source_sample_id = argmax // len(target_data)
target_sample_id = argmax % len(target_data)

# %%
def get_date_positions(dataset, loader):
    folder = os.path.join(config.data_root, dataset)
    meta_folder = os.path.join(folder, "meta")
    metadata = pickle.load(open(os.path.join(meta_folder, "metadata.pkl"), "rb"))
    # dataset dates in format  yyyymmdd (int)
    dates = metadata["dates"]
    # corresponding calendar times
    date_positions = loader.dataset.days_after(metadata["start_date"], dates)
    return dates, date_positions

dates_source, date_positions_source = get_date_positions(config.source[0], source_loader)
dates_target, date_positions_target = get_date_positions(config.target, target_test_loader)

str_dates_source = [str(d) for d in dates_source]
dates_sources = ['-'.join([d[:4], d[4:6], d[6:]]) for d in str_dates_source]

str_dates_target = [str(d) for d in dates_target]
dates_target= ['-'.join([d[:4], d[4:6], d[6:]]) for d in str_dates_target]

# %%
def load_raw_pixels(loader, data, sample_id):
    raw_index = data[sample_id]['index']
    raw_path, _, _, _, _, _ = loader.dataset.samples[raw_index]
    raw_pixels = zarr.load(raw_path)
    return raw_pixels

# %%
raw_source_pixels = load_raw_pixels(source_loader, source_data, source_sample_id)
raw_target_pixels = load_raw_pixels(target_test_loader, target_data, target_sample_id)

# For illustration, we sample a random pixel from the source and target parcels (Each comprised of multiple pixels)
num_pixels_source = raw_source_pixels.shape[-1]
num_pixels_target = raw_target_pixels.shape[-1]
pixel_id_source =  np.random.randint(num_pixels_source)
pixel_id_target =  np.random.randint(num_pixels_target)

raw_source_pixel = raw_source_pixels[:, :, pixel_id_source]
raw_target_pixel = raw_target_pixels[:, :, pixel_id_target]

np.savetxt(f"{config.output_dir}/raw_source_pixel_{config.classes[crop_id]}.csv", raw_source_pixel, delimiter=",")
np.savetxt(f"{config.output_dir}/raw_target_pixel_{region_aliases[TARGET]}_{config.classes[crop_id]}.csv",
            raw_source_pixel, delimiter=",")

combined_data_source = np.vstack((dates_sources, raw_source_pixel.T)).T
np.savetxt(f"{config.output_dir}/raw_source_pixel+dates_{config.classes[crop_id]}.csv", combined_data_source, delimiter=',', fmt='%s')

combined_data_target = np.vstack((dates_target, raw_target_pixel.T)).T
np.savetxt(f"{config.output_dir}/raw_target_pixel+dates_{config.classes[crop_id]}.csv", combined_data_target, delimiter=',', fmt='%s')

# %%
bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates_sources]

for channel in range(raw_source_pixel.shape[1]):
    series = raw_source_pixel[:, channel]
    ax.set_title(f"Input time series of a pixel of class {config.classes[crop_id]}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Reflectance")
    ax.plot(dates, series)

ax.tick_params(axis='x', labelrotation=45)
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
ax.autoscale(tight=True)
ax.legend(bands)
plt.savefig(f"./{config.output_dir}/{EXP}_source_{SOURCE_STR}_{config.classes[crop_id]}.png")
# plt.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

pixel = raw_target_pixel
num_channels = pixel.shape[1]

dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates_target]
for channel in range(num_channels):
    series = pixel[:, channel]
    ax.set_title(f"Input time series of a pixel of class {config.classes[crop_id]}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Reflectance")
    ax.plot(dates, series)

ax.tick_params(axis='x', labelrotation=45)
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
ax.autoscale(tight=True)
ax.legend(bands)
plt.savefig(f"./{config.output_dir}/{EXP}_target_{region_aliases[TARGET]}_{config.classes[crop_id]}.png")
# plt.show()

# %%
red = bands.index('B04')
nir = bands.index('B08')

NDVI = lambda x: (x[:, nir] - x[:, red]) / (x[:, nir] + x[:, red])

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

ndvi_source = NDVI(np.mean(raw_source_pixels, axis=-1))
ndvi_target = NDVI(np.mean(raw_target_pixels, axis=-1))

dt_source = [datetime.strptime(date, '%Y-%m-%d') for date in dates_sources]
dt_target = [datetime.strptime(date, '%Y-%m-%d') for date in dates_target]

plt.plot(dt_source, ndvi_source, label='Source')
plt.plot(dt_target, ndvi_target, label='Target')

ax.set_title(f"NDVI of parcels of class {config.classes[crop_id]}")
ax.set_xlabel("Time")
ax.set_ylabel("NDVI")
ax.tick_params(axis='x', labelrotation=45)
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
ax.autoscale(tight=True)
plt.legend()

plt.savefig(f"./{config.output_dir}/{EXP}_source_NDVI_{config.classes[crop_id]}.png")
# plt.show()


# %%
pe_source = source_data[source_sample_id]['pe']
pe_source = pe_source.T

fig, ax = plt.subplots(1, 1, figsize=(4, 8))
plt.title(f"Positional encoding of a parcel of class {config.classes[crop_id]}")
plt.xlabel("Sequence index t")
plt.ylabel("Dimension")
plt.imshow(pe_source)
plt.colorbar(orientation="horizontal")

plt.savefig(f"./{config.output_dir}/{EXP}_source_pe_{config.classes[crop_id]}.png")


# %%
pe_target = target_data[target_sample_id]['pe']
pe_target = pe_target.T

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
plt.title(f"PE of a parcel of class {config.classes[crop_id]} in the target {region_aliases[TARGET]} tile")
plt.xlabel("Sequence index t")
plt.ylabel("Dimension")
plt.imshow(pe_target)
plt.colorbar(orientation="horizontal")

plt.savefig(f"./{config.output_dir}/{EXP}_target_{region_aliases[TARGET]}_pe_{config.classes[crop_id]}.png")


# %%
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 5))

ax1.set_title(f"PE of a parcel of class {config.classes[crop_id]} in the source {SOURCE_STR} tile")
ax2.set_title(f"PE of a parcel of class {config.classes[crop_id]} in the target {region_aliases[TARGET]} tile")

ax1.set_xlabel("Sequence index t")
ax2.set_xlabel("Sequence index t")
ax1.set_ylabel("Dimension")
ax2.set_ylabel("Dimension")


extent_target = [mdates.date2num(dt_target[0]), mdates.date2num(dt_target[-1]), 0, pe_target.shape[0]]
extent_source = [mdates.date2num(dt_source[0]), mdates.date2num(dt_source[-1]), 0, pe_source.shape[0]]

im1 = ax1.imshow(pe_source, cmap='viridis', aspect='auto', extent=extent_source)
im2 = ax2.imshow(pe_target[:, :30], cmap='viridis', aspect='auto', extent=extent_target)

desired_num_ticks = 15

for ax in (ax1, ax2):
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax.xaxis.set_major_locator(plt.MaxNLocator(desired_num_ticks))

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

fig.colorbar(im2, ax=[ax1, ax2])
out_dir = f"./highlight/{config.experiment_name}"
os.makedirs(out_dir, exist_ok=True)

plt.savefig(f"./{out_dir}/pe_{SOURCE_STR}_{region_aliases[TARGET]}_{config.classes[crop_id]}.png")
plt.savefig(f"./{out_dir}/pe_{SOURCE_STR}_{region_aliases[TARGET]}_{config.classes[crop_id]}.svg")

# %%
fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8, 5))
ax.set_title(f"PE of a parcel of class {config.classes[crop_id]} in the source {region_aliases[TARGET]} tile")
ax.set_xlabel("Time")
ax.set_ylabel("Dimension")


extent = [mdates.date2num(dt_target[0]), mdates.date2num(dt_target[-1]), 0, pe_target.shape[0]]

im1 = ax.imshow(pe_target[:, :30], cmap='viridis', aspect='auto', extent=extent)

ax.xaxis_date()
desired_num_ticks = 15
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
ax.xaxis.set_major_locator(plt.MaxNLocator(desired_num_ticks))

for tick in ax.get_xticklabels():
    tick.set_rotation(45)

fig.colorbar(im1, ax=ax)

out_dir = f"./highlight/{config.experiment_name}"
os.makedirs(out_dir, exist_ok=True)

plt.savefig(f"./{out_dir}/pe_{region_aliases[TARGET]}_{config.classes[crop_id]}.png")
plt.savefig(f"./{out_dir}/pe_{region_aliases[TARGET]}_{config.classes[crop_id]}.svg")
# %%
fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8, 5))


ax.set_title(f"PE of a parcel of class {config.classes[crop_id]} in the source {SOURCE_STR} tile")
ax.set_xlabel("Time")
ax.set_ylabel("Dimension")

extent = [mdates.date2num(dt_source[0]), mdates.date2num(dt_source[-1]), 0, pe_source.shape[0]]

im1 = ax.imshow(pe_target[:, :30], cmap='viridis', aspect='auto', extent=extent)

ax.xaxis_date()
desired_num_ticks = 15
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
ax.xaxis.set_major_locator(plt.MaxNLocator(desired_num_ticks))

for tick in ax.get_xticklabels():
    tick.set_rotation(45)


fig.colorbar(im1, ax=ax)

out_dir = f"./highlight/{config.experiment_name}"
os.makedirs(out_dir, exist_ok=True)

plt.savefig(f"./{out_dir}/pe_{SOURCE_STR}_{config.classes[crop_id]}.png")
plt.savefig(f"./{out_dir}/pe_{SOURCE_STR}_{config.classes[crop_id]}.svg")

# %%
_, raw_warping_paths = dtw_ndim.warping_paths(raw_source_pixel, raw_target_pixel)


fig, ax = dtw_visualisation.plot_warpingpaths(raw_source_pixel, raw_target_pixel, raw_warping_paths, showlegend=True,
                                    s1_title=f"Source pixel series",
                                    s2_title=f"Target pixel series",
                                    )
plt.savefig(f"./{config.output_dir}/{EXP}_warping_path_{config.classes[crop_id]}.png")

# %%
_, ndvi_warping_paths = dtw.warping_paths(ndvi_source, ndvi_target)

fig, ax = dtw_visualisation.plot_warpingpaths(ndvi_source, ndvi_target, ndvi_warping_paths, showlegend=True,
                                    s1_title=f"Source NDVI",
                                    s2_title=f"Target NDVI",
                                    )
plt.savefig(f"./{config.output_dir}/{EXP}_warping_path_NDVI_{config.classes[crop_id]}.png")

# %%
best_path = dtw.best_path(ndvi_warping_paths)
warped_ndvi, path = dtw.warp(ndvi_source, ndvi_target, best_path)
dtw_visualisation.plot_warp(ndvi_source, ndvi_target, warped_ndvi, path)

# %%
_, pe_warping_paths = dtw_ndim.warping_paths(pe_source.T, pe_target.T)
path = dtw.best_path(pe_warping_paths)

fig, ax = plot_warpingpaths(pe_source, pe_target, pe_warping_paths, path, showlegend=True,
                            s1_title=f"Source PE",
                            s2_title=f"Target PE",
                            )
plt.savefig(
    f"./{config.output_dir}/{EXP}_warping_path_pe_{config.classes[crop_id]}.png")

plt.savefig(
    f"./{out_dir}/{EXP}_warping_path_pe_{config.classes[crop_id]}.svg")