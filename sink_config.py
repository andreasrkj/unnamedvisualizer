# Point to where you keep your RADMC runs here
radmc_datadir = "/groups/astro/andreask/production_run" # OLD DIRECTORY '/groups/astro/andreask/radmc'

# Configure the directories in which each sink has its data files
# Format is "XX.YY" with XX being the sink ID and YY being the level, with YY being optional if only one level is considered
sink_dirs = {} # Format tuple(directory, sink_id)
sink_dirs["6"]     = ('/lustre/astro/troels/IMF_512_cores/christian/sink_006/data', 6)
sink_dirs["13"]    = ('/lustre/astro/troels/IMF_512_cores/sink_13/data', 13)
sink_dirs["13.22"] = ('/lustre/astro/troels/IMF_512_cores/sink_13/level_22_resim/data', 13)
sink_dirs["13.24"] = ('/lustre/astro/troels/IMF_512_cores/sink_13/level_24_resim/data', 13)
sink_dirs["24"]    = ('/lustre/astro/troels/IMF_512_cores/christian/sink_025/data', 24)
sink_dirs["82"]    = ('/lustre/astro/troels/IMF_512_cores/christian/sink_082/data', 80)
sink_dirs["122"]   = ('/lustre/astro/troels/IMF_512_cores/christian/sink_122/data', 122)
sink_dirs["162"]   = ('/lustre/astro/troels/IMF_512_cores/christian/sink_162/data', 161)
sink_dirs["180"]   = ('/lustre/astro/troels/IMF_512_cores/christian/sink_180/data', 178)
sink_dirs["225"]   = ('/lustre/astro/troels/IMF_512_cores/christian/sink_225/data', 225)

mesa_dirs = {}
mesa_dirs["6"]     = "/lustre/astro/troels/IMF_512_cores/christian/mesa/sink_006_mesa_data.txt"
mesa_dirs["13"]    = "/lustre/astro/troels/IMF_512_cores/christian/mesa/sink_013_mesa_data.txt"
mesa_dirs["24"]    = "/lustre/astro/troels/IMF_512_cores/christian/mesa/sink_025_mesa_data.txt"
mesa_dirs["82"]    = "/lustre/astro/troels/IMF_512_cores/christian/mesa/sink_082_mesa_data.txt"
mesa_dirs["122"]   = "/lustre/astro/troels/IMF_512_cores/christian/mesa/sink_122_mesa_data.txt"
mesa_dirs["162"]   = "/lustre/astro/troels/IMF_512_cores/christian/mesa/sink_162_mesa_data.txt"
mesa_dirs["180"]   = "/lustre/astro/troels/IMF_512_cores/christian/mesa/sink_180_mesa_data.txt"
mesa_dirs["225"]   = "/lustre/astro/troels/IMF_512_cores/christian/mesa/sink_225_mesa_data.txt"