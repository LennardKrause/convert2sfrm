# BioMAX2sfrm, version 1.0.0 (released 15.04.2025)
Convert BioMAX (EIGER2) or MicroMAX (EIGER2/JUNGFRAU) data to Bruker .sfrm format

## How to use
### General
 - Drag and drop the h5 master files (_*_master.h5_) onto the window
 - You can drop all files, only _*_master.h5_ files will be added
 - You can change the run number and the kappa angle for each run
 - The rows (runs) will be converted successively using all the cores you have

### Buttons
 - Click **Convert** to start the conversion
 - Using **Stop** might save storage space by aborting the process
 - The **X** button will remove the row from the table
 - Untick **Overwrite** if you want to save time finishing previously aborted conversions

### Features
 - If multiple _*_master.h5_ files are dropped together,<br> they have their run number and kappa angle incremented automatically


### FUTURE
 - figure out a way to calculate the sensor efficiency fast (without imports) we know energy, material and thickness
 - Bruker/APEX doesn't support the JUNGFRAU detector so we tell it to use the EIGER settings e.g. HPAD, 10/37 px gaps by settings the name to: 'JUNGFRAU(EIGER2)' APEX only parses for 'EIGER' in the name
