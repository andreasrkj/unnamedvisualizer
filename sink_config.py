# Configure the directories in which each sink has its data files
# Format is "XX.YY" with XX being the sink ID and YY being the level, with YY being optional if only one level is considered
sink_dirs = {}
sink_dirs["13"] = '/lustre/astro/troels/IMF_512_cores/sink_13/data'
sink_dirs["13.22"] = '/lustre/astro/troels/IMF_512_cores/sink_13/level_22_resim/data'
sink_dirs["13.24"] = '/lustre/astro/troels/IMF_512_cores/sink_13/level_24_resim/data'
sink_dirs["24"]    = '/lustre/astro/troels/IMF_512_cores/christian/sink_025/data'

# Point to where you keep your RADMC runs here
radmc_datadir = '/groups/astro/andreask/radmc'