
# Read in calibration parameters for MA model
calibration_params <- readRDS("data/cal_res_MA/cal_samp/Calib_par_table.rds")

# Read in calibration results table for MA model
file_path <- "data/cal_res_MA/cal_res/"
filelist <- list.files(file_path, '*.rds')
filelist <- paste0(file_path, filelist)
datasets <- lapply(filelist, readRDS)
calibration_results <- do.call(rbind, datasets)
rm(datasets)

# Combine get ready for BayCann
cal_res <- cbind(calibration_results, t(calibration_params[, calibration_results[, "index"]]))
saveRDS(cal_res, file = "data/cal_res_MA/cal_res.RDS")
