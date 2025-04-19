# convert2sfrm, version 1.0.2 (released 19.04.2025)
An easy-to-use GUI to convert [BioMAX](https://www.maxiv.lu.se/beamlines-accelerators/beamlines/biomax/) ([EIGER2](https://www.dectris.com/en/detectors/x-ray-detectors/eiger2/)) or [MicroMAX](https://www.maxiv.lu.se/beamlines-accelerators/beamlines/micromax/) ([EIGER2](https://www.dectris.com/en/detectors/x-ray-detectors/eiger2/)/[JUNGFRAU](https://www.psi.ch/en/lxn/jungfrau)) data to the [Bruker](https://www.bruker.com) ***.sfrm*** format. It makes extensive use of hardcoded defaults to streamline the experience. If you are looking to convert [PILATUS3](https://www.dectris.com/en/support/manuals-docs/pilatus3-x-cdte-for-synchrotron/) data, I have [good news](https://github.com/LennardKrause/p3fc) for you too!

>[!IMPORTANT]
> This is work in progress, please make sure to reach out if you run into trouble.
> 
> New build released, please read below.

![Example picture of the conversion window.](/assets/BioMAX2sfrm.png)

## How to use
### General
 - Download `convert2sfrm-1.0.2-py3-none-any.whl`
 - Install using `python3 -m pip install /path/to/convert2sfrm-1.0.2-py3-none-any.whl`
 - Run by typing `BioMAX2sfrm` in a terminal
 - Drag and drop the h5 master files (****_master.h5***) onto the window
 - You can drop all files, only ****_master.h5*** files will be added
 - You can change the run number and the kappa angle for each run
 - The rows (runs) will be converted successively using all the cores you have
 - The frames will be converted into the ****_sfrm*** folder

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
 - Bruker/APEX doesn't support the JUNGFRAU detector so we tell it to use the EIGER settings e.g. HPAD, 10/37 px gaps by setting the name to: 'JUNGFRAU(EIGER2)' as APEX only parses for 'EIGER' in the name and the JUNGFRAU bases on the EIGER2 layout

### I hope this turns out to be useful for someone!
