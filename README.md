# BioMAX2sfrm, version 1.0.0 (released 15.04.2025)
Convert BioMAX (EIGER2) or MicroMAX (EIGER2/JUNGFRAU) data to Bruker .sfrm format

>[!IMPORTANT]
> This is work in progress, please make sure to reach out if you run into troubles

## How to use
### General
 - Drag and drop the h5 master files (****_master.h5***) onto the window
 - You can drop all files, only ****_master.h5*** files will be added
 - You can change the run number and the kappa angle for each run
 - The rows (runs) will be converted successively using all the cores you have

### Buttons
 - Click **Convert** to start the conversion
 - Using **Stop** might save storage space by aborting the process
 - The **X** button will remove the row from the table
 - Untick **Overwrite** if you want to save time finishing previously aborted conversions

### Features
 - If multiple ****_master.h5*** files are dropped together, they have their run number and kappa angle incremented automatically


### Future
 - Once the kappa angle is stored in the h5 the angle will be set automatically
 - Figure out a way to calculate the sensor efficiency fast (without imports) we know energy, material and thickness
 - Bruker/APEX doesn't support the JUNGFRAU detector so we tell it to use the EIGER settings e.g. HPAD, 10/37 px gaps by setting the name to: 'JUNGFRAU(EIGER2)' as APEX only parses for 'EIGER' in the name

### I hope this turns out to be useful for someone!
