import os
import collections
import numpy as np
import h5py, hdf5plugin

def prepare_saint_mask(h5file, h5path, par):
    # open the h5 file and read the first image
    with h5py.File(h5file, 'r') as h5:
        data = h5[h5path][0,:,:]
    
    # make up a name for the mask
    # the masks we use are called X-ray Aperture (xa) files in saint and
    # have the naming convention: some_data_xa_01_0001.sfrm 
    # they can be provided per run and must be stored together with the frames
    mask_name = '{}_xa_{:02}_0001.sfrm'.format(os.path.join(par['frm_path'], par['frm_name']), par['frm_run'])
    
    # the dead areas are flagged as saturated, so their value is 65535 for
    # 16 bit unsigned integers. Eiger2 images can be stored at 16 or 32 bit
    # par['det_max_counts'] knows that value.
    # Everything saturated can't be trusted, everything below (even zero)
    # is data and is set to 1 (active)
    data[data < par['det_max_counts']] = 1
    # the dead areas are set to 0 (inactive)
    data[data == par['det_max_counts']] = 0
    data = data.astype(np.uint8)
    
    # calculate detector pixel per cm
    # this is normalized to a 512x512 detector format
    # Eiger2-1M pixel size is 0.075 mm 
    pix_per_512 = round((10.0 / 0.075) * (512.0 / data.shape[0]), 6)
    
    # default Bruker header
    header = init_bruker_header()
    
    # fill known header items
    header['NCOLS']      = [data.shape[1], 4]
    header['NROWS']      = [data.shape[0], 8]
    header['CCDPARM']    = [0.0, 1.00, 1.00, 0.0, (2**20)*par['frm_sum']]
    header['DETPAR']     = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    header['DETTYPE']    = ['Eiger2', pix_per_512, 0.0, 0, 0.0, 0.0012, 1]
    header['SITE']       = par['src_facility_name']
    header['MODEL']      = par['src_beamline_name']
    header['TARGET']     = par['src_facility_type']
    header['WAVELEN']    = [par['src_wavelength'],par['src_wavelength'],par['src_wavelength']]
    header['TYPE']       = 'ACTIVE MASK'
    header['NPIXELB']    = [1, 1]
    header['MAXIMUM']    = 1
    header['MINIMUM']    = 0
    header['NCOUNTS']    = [data.sum(), 0]
    header['PHD']        = [1.00, 0.1]
    
    # write the frame
    write_bruker_frame(mask_name, header, data)

def init_bruker_header():
    header = collections.OrderedDict()
    header['FORMAT']  = 100                                                     # Frame Format -- 86=SAXI, 100=Bruker
    header['VERSION'] = 18                                                      # Header version number
    header['HDRBLKS'] = 15                                                      # Header size in 512-byte blocks
    header['TYPE']    = '?'                                                     # String indicating kind of data in the frame
    header['SITE']    = '?'                                                     # Site name
    header['MODEL']   = '?'                                                     # Diffractometer model
    header['USER']    = '?'                                                     # Username
    header['SAMPLE']  = '?'                                                     # Sample ID
    header['SETNAME'] = '?'                                                     # Basic data set name
    header['RUN']     = 1                                                       # Run number within the data set
    header['SAMPNUM'] = 1                                                       # Specimen number within the data set
    header['TITLE']   = ['', '', '', '', '', '', '', '']                        # User comments (8 lines)
    header['NCOUNTS'] = np.array([-9999, 0])                                    # Total frame counts, Reference detector counts
    header['NOVERFL'] = np.array([-1, 0, 0])                                    # SAXI Format: Number of overflows
                                                                                # Bruker Format: #Underflows; #16-bit overfl; #32-bit overfl
    header['MINIMUM'] = 0                                                       # Minimum pixel value
    header['MAXIMUM'] = 1                                                       # Maximum pixel value
    header['NONTIME'] = 0                                                       # Number of on-time events
    header['NLATE']   = 0                                                       # Number of late events for multiwire data
    header['FILENAM'] = '?'                                                     # (Original) frame filename
    header['CREATED'] = '?'                                                     # Date and time of creation
    header['CUMULAT'] = 0.0                                                     # Accumulated exposure time in real hours
    header['ELAPSDR'] = 0.0                                                     # Requested time for this frame in seconds
    header['ELAPSDA'] = 0.0                                                     # Actual time for this frame in seconds
    header['OSCILLA'] = 0                                                       # Nonzero if acquired by oscillation
    header['NSTEPS']  = 0                                                       # steps or oscillations in this frame
    header['RANGE']   = 0.0                                                     # Magnitude of scan range in decimal degrees
    header['START']   = 0.0                                                     # Starting scan angle value, decimal deg
    header['INCREME'] = 0.0                                                     # Signed scan angle increment between frames
    header['NUMBER']  = 1                                                       # Number of this frame in series (zero-based)
    header['NFRAMES'] = 1                                                       # Number of frames in the series
    header['ANGLES']  = np.array([0.0, 0.0, 0.0, 0.0])                          # Diffractometer setting angles, deg. (2Th, omg, phi, chi)
    header['NOVER64'] = 0                                                       # Number of pixels > 64K
    header['NPIXELB'] = np.array([1, 1])                                        # Number of bytes/pixel; Number of bytes per underflow entry
    header['NROWS']   = np.array([512, 1])                                      # Number of rows in frame; number of mosaic tiles in Y; dZ/dY value
                                                                                # for each mosaic tile, X varying fastest
    header['NCOLS']   = np.array([512, 1])                                      # Number of pixels per row; number of mosaic tiles in X; dZ/dX
                                                                                # value for each mosaic tile, X varying fastest
    header['WORDORD'] = 0                                                       # Order of bytes in word; always zero (0=LSB first)
    header['LONGORD'] = 0                                                       # Order of words in a longword; always zero (0=LSW first
    header['TARGET']  = '?'                                                     # X-ray target material)
    header['SOURCEK'] = 0.0                                                     # X-ray source kV
    header['SOURCEM'] = 0.0                                                     # Source milliamps
    header['FILTER']  = '?'                                                     # Text describing filter/monochromator setting
    header['CELL']    = np.array([1.0, 1.0, 1.0, 90.0, 90.0, 90.0])             # Cell constants, 2 lines  (A,B,C,Alpha,Beta,Gamma)
    header['MATRIX']  = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]) # Orientation matrix, 3 lines
    header['LOWTEMP'] = np.array([1, -17300, -6000])                            # Low temp flag; experiment temperature*100; detector temp*100
    header['ZOOM']    = np.array([0.0, 0.0, 1.0])                               # Image zoom Xc, Yc, Mag
    header['CENTER']  = np.array([256.0, 256.0, 256.0, 256.0])                  # X, Y of direct beam at 2-theta = 0
    header['DISTANC'] = 5.0                                                     # Sample-detector distance, cm
    header['TRAILER'] = -1                                                      # Byte pointer to trailer info (unused; obsolete)
    header['COMPRES'] = 0                                                       # Text describing compression method if any
    header['LINEAR']  = np.array([1.0, 0.0])                                    # Linear scale, offset for pixel values
    header['PHD']     = np.array([1.0, 0.1])                                    # Discriminator settings
    header['PREAMP']  = 0                                                       # Preamp gain setting
    header['CORRECT'] = 'INTERNAL'                                              # Flood correction filename
    header['WARPFIL'] = 'LINEAR'                                                # Spatial correction filename
    header['WAVELEN'] = np.array([0.0, 0.0, 0.0])                               # Wavelengths (average, a1, a2)
    header['MAXXY']   = np.array([0, 0])                                        # X,Y pixel # of maximum counts
    header['AXIS']    = 2                                                       # Scan axis (1=2-theta, 2=omega, 3=phi, 4=chi)
    header['ENDING']  = np.array([0.0, 0.0, 0.0, 0.0])                          # Setting angles read at end of scan
    header['DETPAR']  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])                # Detector position corrections (Xc,Yc,Dist,Pitch,Roll,Yaw)
    header['LUT']     = 'lut'                                                   # Recommended display lookup table
    header['DISPLIM'] = np.array([0.0, 63.0])                                   # Recommended display contrast window settings
    header['PROGRAM'] = 'Python Image Conversion'                               # Name and version of program writing frame
    header['ROTATE']  = 0                                                       # Nonzero if acquired by rotation (GADDS)
    header['BITMASK'] = '$NULL'                                                 # File name of active pixel mask (GADDS)
    header['OCTMASK'] = np.array([0, 0, 0, 0, 0, 0, 0, 0])                      # Octagon mask parameters (GADDS) #min x, min x+y, min y, max x-y, max x, max x+y, max y, max y-x
    header['ESDCELL'] = np.array([0.001, 0.001, 0.001, 0.02, 0.02, 0.02])       # Cell ESD's, 2 lines (A,B,C,Alpha,Beta,Gamma)
    header['DETTYPE'] = 'UNKNOWN'                                               # Detector type
    header['NEXP']    = np.array([1, 0, 0, 0, 0])                               # Number exposures in this frame; CCD bias level*100,;
                                                                                # Baseline offset (usually 32); CCD orientation; Overscan Flag
    header['CCDPARM'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])                     # CCD parameters for computing pixel ESDs; readnoise, e/ADU, e/photon, bias, full scale
    header['CHEM']    = '?'                                                     # Chemical formula
    header['MORPH']   = '?'                                                     # CIFTAB string for crystal morphology
    header['CCOLOR']  = '?'                                                     # CIFTAB string for crystal color
    header['CSIZE']   = '?'                                                     # String w/ 3 CIFTAB sizes, density, temp
    header['DNSMET']  = '?'                                                     # CIFTAB string for density method
    header['DARK']    = 'INTERNAL'                                              # Dark current frame name
    header['AUTORNG'] = np.array([0.0, 0.0, 0.0, 0.0, 1.0])                     # Autorange gain, time, scale, offset, full scale
    header['ZEROADJ'] = np.array([0.0, 0.0, 0.0, 0.0])                          # Adjustments to goniometer angle zeros (tth, omg, phi, chi)
    header['XTRANS']  = np.array([0.0, 0.0, 0.0])                               # Crystal XYZ translations
    header['HKL&XY']  = np.array([0.0, 0.0, 0.0, 0.0, 0.0])                     # HKL and pixel XY for reciprocal space (GADDS)
    header['AXES2']   = np.array([0.0, 0.0, 0.0, 0.0])                          # Diffractometer setting linear axes (4 ea) (GADDS)
    header['ENDING2'] = np.array([0.0, 0.0, 0.0, 0.0])                          # Actual goniometer axes @ end of frame (GADDS)
    header['FILTER2'] = np.array([0.0, 0.0, 0.0, 1.0])                          # Monochromator 2-theta (deg), roll (deg), beam tilt, attenuation factor
    header['LEPTOS']  = ''
    #header['CFR']     = ['']
    return header

def write_bruker_frame(fname, fheader, fdata):
    ########################
    ## write_bruker_frame ##
    ##     FUNCTIONS      ##
    ########################
    def pad_table(table, bpp):
        '''
        pads a table with zeros to a multiple of 16 bytes
        '''
        padded = np.zeros(int(np.ceil(table.size * abs(bpp) / 16)) * 16 // abs(bpp)).astype(_BPP_TO_DT[bpp])
        padded[:table.size] = table
        return padded

    def format_bruker_header(fheader):
        '''
        
        '''
        fmt = {1:'{:<71} ',
               2:'{:<35} {:<35} ',
               3:'{:<23} {:<23} {:<23} ',
               4:'{:<17} {:<17} {:<17} {:<17} ',
               5:'{:<13} {:<13} {:<13} {:<13} {:<15} '}
    
        hdr_lines = []
        for name, entry in fheader.items():
            # cast entry to array
            entry = np.atleast_1d(entry)
            
            # round if (all) float
            if np.issubdtype(entry.dtype, np.floating):
                entry = np.round(entry, 6)

            # store header block index
            # calculate / write number
            # of header blocks at the end
            if name == 'HDRBLKS':
                hdr_hdrblks_idx = len(hdr_lines)

            # TITLE has 8 lines
            if name == 'TITLE':
                num = len(entry)
                for idx in range(8):
                    if num < idx:
                        line = f'{name:<7}:{entry[idx]:<72}'
                    else:
                        line = f'{name:<7}:{" ":<72}'
                    hdr_lines.append(line)
                    assert len(line) == 80, f'Error writing Bruker header, line exceeds 80 characters! Key: {name}'
                continue
            
            # OCTMASK: 6+2 entries in line
            if name == 'OCTMASK':
                part = '{:<11} {:<11} {:<11} {:<11} {:<11} {:<11} '.format(*entry[:6])
                assert len(part) == 72, f'Error writing Bruker header, line exceeds 80 characters! Key: {name}'
                hdr_lines.append(f'{name:<7}:{part}')
                part = fmt[len(entry[6:])].format(*entry[6:])
                assert len(part) == 72, f'Error writing Bruker header, line exceeds 80 characters! Key: {name}'
                hdr_lines.append(f'{name:<7}:{part}')
                continue
            
            # DETTYPE: Mixes Entry Types
            if name == 'DETTYPE':
                part = '{:<20} {:<11} {:<10} {:<2} {:<10} {:<10} {:<2} '.format(*entry)
                assert len(part) == 72, f'Error writing Bruker header, line exceeds 80 characters! Key: {name}'
                hdr_lines.append(f'{name:<7}:{part}')
                continue
            
            # write entries, up to 5 per line
            part = fmt[len(entry[:5])].format(*entry[:5])
            assert len(part) == 72, f'Error writing Bruker header, line exceeds 80 characters! Key: {name}'
            hdr_lines.append(f'{name:<7}:{part}')
            
            # add new line for remainder if more than 5
            if len(entry) > 5:
                part = fmt[len(entry[5:])].format(*entry[5:])
                assert len(part) == 72, f'Error writing Bruker header, line exceeds 80 characters! Key: {name}'
                hdr_lines.append(f'{name:<7}:{part}')
            continue
    
        # add header ending
        # header must be multiple of 512 bytes
        # 1 header block (HDRBLKS) is 512 bytes
        hdr_pad = 512 - (len(hdr_lines) * 80 % 512)
        hdr_end = '\x1a\x04'          # 2 bytes
        hdr_last = 'CFR: HDR: IMG: ' # 15 bytes
        
        # fill to next 512 bytes
        while hdr_pad > 80:
            hdr_lines.append(hdr_end + ''.join(['.'] * 78))
            hdr_pad -= 80
        
        # close header
        hdr_fill = str('.' * (hdr_pad - 17))
        hdr_lines.append(hdr_last + hdr_fill + hdr_end)
        
        # calculate new header size
        # and update header blocks
        name = 'HDRBLKS'
        entry = ((len(hdr_lines)-1) * 80 + hdr_pad) // 512
        hdr_lines[hdr_hdrblks_idx] = f'{name:<7}:{entry:<72}'

        return ''.join(hdr_lines)
    ########################
    ## write_bruker_frame ##
    ##   FUNCTIONS END    ##
    ########################
    
    # assign bytes per pixel to numpy integers
    # int8   Byte (-128 to 127)
    # int16  Integer (-32768 to 32767)
    # int32  Integer (-2147483648 to 2147483647)
    # uint8  Unsigned integer (0 to 255)
    # uint16 Unsigned integer (0 to 65535)
    # uint32 Unsigned integer (0 to 4294967295)
    _BPP_TO_DT = {1: np.uint8,
                  2: np.uint16,
                  4: np.uint32,
                 -1: np.int8,
                 -2: np.int16,
                 -4: np.int32}
    
    # read the bytes per pixel
    # frame data (bpp), underflow table (bpp_u)
    bpp, bpp_u = fheader['NPIXELB']

    # expand data to int64
    fdata = fdata.astype(np.int64)
    
    # generate underflow table
    # does not work as APEXII reads the data as uint8/16/32!
    if fheader['NOVERFL'][0] >= 0:
        data_underflow = fdata[fdata <= 0]
        fheader['NOVERFL'][0] = data_underflow.shape[0]
        table_underflow = pad_table(data_underflow, -1 * bpp_u)
        fdata[fdata < 0] = 0

    # generate 32 bit overflow table
    if bpp < 4:
        data_over_uint16 = fdata[fdata >= 65535]
        table_data_uint32 = pad_table(data_over_uint16, 4)
        fheader['NOVERFL'][2] = data_over_uint16.shape[0]
        fdata[fdata >= 65535] = 65535

    # generate 16 bit overflow table
    if bpp < 2:
        data_over_uint8 = fdata[fdata >= 255]
        table_data_uint16 = pad_table(data_over_uint8, 2)
        fheader['NOVERFL'][1] = data_over_uint8.shape[0]
        fdata[fdata >= 255] = 255

    # shrink data to desired bpp
    fdata = fdata.astype(_BPP_TO_DT[bpp])
    
    # write frame
    with open(fname, 'wb') as brukerFrame:
        brukerFrame.write(format_bruker_header(fheader).encode('ASCII'))
        brukerFrame.write(fdata.tobytes())
        if fheader['NOVERFL'][0] >= 0:
            brukerFrame.write(table_underflow.tobytes())
        if bpp < 2 and fheader['NOVERFL'][1] > 0:
            brukerFrame.write(table_data_uint16.tobytes())
        if bpp < 4 and fheader['NOVERFL'][2] > 0:
            brukerFrame.write(table_data_uint32.tobytes())

def kappa_to_euler(k_omg, kappa, alpha, k_phi):
    '''
     converts kappa to eulerian geometry
     needed for .sfrm header
     r_: in radians
     k_: kappa geometry
     e_: euler geometry
     alternative delta:
     r_delta_ = np.arcsin(np.cos(np.deg2rad(alpha)) * np.sin(np.deg2rad(kappa) / 2.0) / np.cos(r_e_chi / 2.0))
    '''
    r_k_omg = np.deg2rad(k_omg)
    r_k_phi = np.deg2rad(k_phi)
    r_e_chi = 2.0 * np.arcsin(np.sin(np.deg2rad(kappa) / 2.0) * np.sin(np.deg2rad(alpha)))
    r_delta = np.arccos(np.cos(np.deg2rad(kappa) / 2.0) / np.cos(r_e_chi / 2.0))
    e_omg = np.round(np.rad2deg(r_k_omg + r_delta), 5)
    e_phi = np.round(np.rad2deg(r_k_phi + r_delta), 5)
    e_chi = np.round(np.rad2deg(r_e_chi), 5)
    return e_omg, e_phi, e_chi

def convert_h5_to_sfrm(h5_path, h5_idx, frm_num, par):
    frm_name = '{}_{:02}_{:04}.sfrm'.format(os.path.join(par['frm_path'], par['frm_name']), par['frm_run'], frm_num)
    # Overwrite existing files if overwrite is flagged True
    if not par['overwrite'] and os.path.isfile(frm_name):
        return False
        
    # open the h5 file and sum the images in the given range
    with h5py.File(par['h5_file'], 'r') as h5f:
        frm_data = np.sum(h5f[h5_path][h5_idx:h5_idx+par['frm_sum'],:,:], axis=0)

    # set bad pixels to zero
    # this is only cosmetic but makes the indexing / visual analysis easier
    # as long as the xa mask files are used during integration
    # these pixels are masked and properly handled with
    frm_data[frm_data == par['det_max_counts'] * par['frm_sum']] = 0
    frm_data[frm_data < 0] = 0

    # scan parameters
    # increment, exposure time, start and end angle of the omega scan
    frm_idx = (frm_num - 1) * par['frm_sum']
    scn_inc = par['gon_scan_width'] * par['frm_sum']
    scn_exp = par['det_frame_time'] * par['frm_sum']
    scn_sta = par['gon_omega_start'][frm_idx]
    scn_end = par['gon_omega_end'][frm_idx + par['frm_sum'] - 1]

    # convert kappa to euler geometry
    omg_0, phi_0, chi_0 = kappa_to_euler(scn_sta, par['kappa'], par['alpha'], 0)
    omg_1, phi_1, chi_1 = kappa_to_euler(scn_end, par['kappa'], par['alpha'], 0)

    # angle correction
    phi_0 = 0.0 - phi_0
    phi_1 = 0.0 - phi_1
    omg_0 = omg_0 - 40.8
    omg_1 = omg_1 - 40.8

    # calculate detector pixel per cm
    # this is normalized to a 512x512 detector format
    # Eiger2 pixel size is 0.075 mm
    pix_per_512 = round((10.0 / (par['det_pxs']*1e3)) * (512.0 / frm_data.shape[0]), 6)
    
    # default bruker header
    header = init_bruker_header()
    
    # fill known header items
    header['NCOLS']      = [frm_data.shape[1], 4]                               # Number of pixels per row; number of mosaic tiles in X; dZ/dX
    header['NROWS']      = [frm_data.shape[0], 8]                               # Number of rows in frame; number of mosaic tiles in Y; dZ/dY value
    header['CENTER']     = [par['det_bc_x'], par['det_bc_y'],
                            par['det_bc_x'], par['det_bc_y']]                   # adjust the beam center for the filling/cutting of the frame
    header['DETTYPE']    = [par['det_name'], pix_per_512, 0.001, 0, 0.001, 0.001, 1]   # Detector type, pix512percm, cmtogrid, circular, brassspacing, windowthickness, accuratetiming
    header['DETPAR']     = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]                       # Detector position corrections (Xc,Yc,Dist,Pitch,Roll,Yaw)
    header['CCDPARM']    = [0.0, 1.0, 1.0, 0.0, (2**20)*par['frm_sum']]         # CCD parameters for computing pixel ESDs; readnoise, e/ADU, e/photon, bias, full scale
    header['SITE']       = par['src_facility_name']                             # Site name
    header['MODEL']      = par['src_beamline_name']                             # Diffractometer model
    header['TARGET']     = par['src_facility_type']                             # X-ray target material)
    header['USER']       = par['user_name']                                     # Username
    header['FILENAM']    = par['frm_name']                                      # (Original) frame filename
    header['RUN']        = par['frm_run']                                       # Run number within the data set
    header['SAMPLE']     = par['sample_name']                                   # Samplename
    header['SAMPNUM']    = 0                                                    # Specimen number within the data set
    header['SETNAME']    = '?'                                                  # Setname
    header['FILTER']     = 'Si(111) DCM (hor.), KB mirror pair (VFM, HFM)'      # Text describing filter/monochromator setting
    header['SOURCEK']    = '?'                                                  # X-ray source kV
    header['SOURCEM']    = '?'                                                  # Source milliamps
    header['WAVELEN']    = [par['src_wavelength'],par['src_wavelength'],par['src_wavelength']] # Wavelengths (average, a1, a2)
    header['CUMULAT']    = scn_exp                                              # Accumulated exposure time in real hours
    header['ELAPSDR']    = scn_exp                                              # Requested time for this frame in seconds
    header['ELAPSDA']    = scn_exp                                              # Actual time for this frame in seconds
    header['START']      = omg_0                                                # Starting scan angle value, decimal deg
    header['ANGLES']     = [0.0, omg_0, phi_0, chi_0]                           # Diffractometer setting angles, deg. (2Th, omg, phi, chi)
    header['ENDING']     = [0.0, omg_1, phi_1, chi_1]                           # Setting angles read at end of scan
    header['TYPE']       = 'Generic Omega Scan'                                 # String indicating kind of data in the frame
    header['DISTANC']    = par['det_distance'] * 100                            # Sample-detector distance, cm
    header['RANGE']      = abs(scn_inc)                                         # Magnitude of scan range in decimal degrees
    header['INCREME']    = scn_inc                                              # Signed scan angle increment between frames
    header['NUMBER']     = frm_num                                              # Number of this frame in series (zero-based)
    header['NFRAMES']    = len(par['gon_omega_start'])                          # Number of frames in the series
    header['AXIS']       = 2                                                    # Scan axis (1=2-theta, 2=omega, 3=phi, 4=chi)
    header['LOWTEMP']    = [1, int((-273.15 + 100.0) * 100.0), -6000]           # Low temp flag; experiment temperature*100; detector temp*100
    header['NEXP']       = [1, 0, 0, 0, 0]                                      # Number exposures in this frame; CCD bias level*100,;
                                                                                # Baseline offset (usually 32); CCD orientation; Overscan Flag
    header['MAXXY']      = np.vstack((frm_data == frm_data.max()).nonzero())[:,0] # X,Y pixel # of maximum counts
    header['MAXIMUM']    = np.max(frm_data)
    header['MINIMUM']    = np.min(frm_data)
    header['NCOUNTS']    = [frm_data.sum(), 0]
    header['NOVER64']    = frm_data[frm_data > 64000].shape[0]                  # Number of pixels > 64K
    header['PHD']        = [par['sensor_efficiency'], par['sensor_thickness']*1e2] # Phosphor efficiency, phosphor thickness [cm]
    ox = frm_data.shape[1]
    oy = frm_data.shape[0]
    header['OCTMASK']    = [0, 0, 0, ox-1, ox-1, ox+oy-1, oy-1, oy-1]           # Octagon mask parameters (GADDS) #min x, min x+y, min y, max x-y, max x, max x+y, max y, max y-x
    header['FILTER2']    = [90.0, 90.0, 0.0, 1.0]                               # Monochromator 2-theta (deg), roll (deg), beam tilt, attenuation factor
    # Monochromator 2-theta (deg) : Degree of polarisation, 0.0 unpolarised, 90.0 fully polarised
    # Monochromator roll (deg)    : Direction of the plane of polarization
    # Synchrotron, goniometer flat: 90.0 90
    # Synchrotron, goniometer side: 90.0  0
    # Lab-Source , mirror         :  0.0  0
    
    # write the frame
    write_bruker_frame(frm_name, header, frm_data)
    return True

def prepare_BioMAX(h5_file, run_number, gon_kappa):
    # BioMAX Dectris EIGER2 CdTe 16M D023193
    h5_paths = {'dat_images':'entry/data/',
                'src_beamline_name':'entry/instrument/name',
                'src_facility_name':'entry/source/name',
                'src_facility_type':'entry/source/type',
                'src_wavelength':'entry/instrument/beam/incident_wavelength',
                'det_bc_x':'entry/instrument/detector/beam_center_x',
                'det_bc_y':'entry/instrument/detector/beam_center_y',
                'det_distance':'entry/instrument/detector/detector_distance',
                'det_frame_time':'entry/instrument/detector/frame_time',
                'det_bit_depth':'entry/instrument/detector/bit_depth_image',
                'det_id':'entry/instrument/detector/detector_number',
                'det_desc':'entry/instrument/detector/description',
                'det_pxs_x':'entry/instrument/detector/x_pixel_size',
                'det_pxs_y':'entry/instrument/detector/y_pixel_size',
                'det_px_x':'entry/instrument/detector/detectorSpecific/x_pixels_in_detector',
                'det_px_y':'entry/instrument/detector/detectorSpecific/y_pixels_in_detector',
                'sensor_material':'entry/instrument/detector/sensor_material',
                'sensor_thickness':'entry/instrument/detector/sensor_thickness',
                'gon_scan_width':'entry/sample/goniometer/omega_range_average',
                'gon_omega_start':'entry/sample/goniometer/omega',
                'gon_omega_end':'entry/sample/goniometer/omega_end',
                'gon_omega':'entry/sample/transformations/Omega',
                }

    par = {}
    par['h5_file'] = h5_file

    with h5py.File(par['h5_file'], 'r') as h5f:
        for key, path in h5_paths.items():
            if key == 'dat_images':
                par[key] = list()
                for end in list(h5f[path].keys()):
                    link = os.path.join(path, end)
                    # check if the files are available or None
                    if h5f.get(link, default=None) is not None:
                        par[key].append({'link':link, 'num':len(h5f[link])})
            elif path in h5f:
                value = h5f[path][()]
                if isinstance(value, np.bytes_):
                    par[key] = value.decode()
                elif isinstance(value, float) or isinstance(value, np.floating):
                    par[key] = np.round(float(value), 6)
                elif isinstance(value, int) or isinstance(value, np.integer):
                    par[key] = int(value)
                else:
                    par[key] = value

    assert par['det_pxs_x'] == par['det_pxs_y'], 'Pixel sizes are not equal!'
    par['det_pxs'] = par['det_pxs_x']

    if 'JUNGFRAU' in par['det_desc']:
        par['gon_scan_width'] = np.round(np.mean(par['gon_omega'][1:]-par['gon_omega'][:-1]), 6)
        par['gon_omega_start'] = np.round(par['gon_omega'], 6)
        par['gon_omega_end'] = np.round(np.append(par['gon_omega'][1:], np.array([par['gon_omega'][-1]+par['gon_scan_width']])), 6)
        # Bruker/APEX doesn't support the JUNGFRAU detector
        # so we tell it to use the EIGER settings
        # e.g. HPAD, 10/37 px gaps
        # APEX only parses for EIGER in the name
        par['det_name'] = 'JUNGFRAU(EIGER2)'
        # *** FUTURE ***
        # figure out a way to calculate the efficiency fast (without imports)
        # we know energy, material and thickness
        par['sensor_efficiency'] = 1.0
    elif 'EIGER2' in par['det_desc']:
        par['det_name'] = 'EIGER2'
        par['sensor_efficiency'] = 1.0
    else:
        print('Detector not supported!')
        return

    par['user_name'] = '?'
    par['sample_name'] = '?'
    par['sample_number'] = 0
    
    temp_path, temp_name = os.path.split(par['h5_file'])
    # get run number from h5-file name
    # e.g. sample1-crystal2_3_master.h5
    #  - remove the extension
    #  - remove '_master'
    #  - reverse the string
    #  - split once at '_'
    par['frm_run'], par['frm_name'] = (os.path.splitext(temp_name)[0]).removesuffix('_master')[::-1].split('_', 1)
    par['frm_name'] = par['frm_name'][::-1]
    par['frm_run'] = int(run_number)
    # sum images to 0.5 degree scan frames
    #  - save storage space
    #  - frame queue in SAINT is memory limited (> EIGER2 16M)
    par['frm_sum'] = max(int(round(0.5 / par['gon_scan_width'], 0)), 1)
    # mini-kappa angle in degrees
    par['kappa'] = float(gon_kappa)
    # mini-kappa alpha angle in degrees
    par['alpha'] = 24.0
    # target sfrm directory
    par['frm_path'] = os.path.join(temp_path, f'{par["frm_name"]}_sfrm')
    # get max vlaue (16 or 32 bit images) to flag dead pixels
    par['det_max_counts'] = 2**par['det_bit_depth']-1
    # total number of images in h5-file
    par['dat_images_num'] = sum([i['num'] for i in par['dat_images']])
    # Bruker convention
    # detector [0,0] is lower left
    # numpy [0,0] is upper left
    par['det_bc_y'] = par['det_px_y'] - par['det_bc_y']
    
    # create the output directory
    if not os.path.exists(par['frm_path']):
        os.mkdir(par['frm_path'])
    
    # prepare a SAINT integration mask (X-ray Aperture, xa) file
    prepare_saint_mask(par['h5_file'], par['dat_images'][0]['link'], par)

    # todo: total number of images to be converted
    assert par['dat_images_num'] % par['frm_sum'] == 0, 'Number of images must be an integer multiple of image summation!'
    progress_total = par['dat_images_num'] // par['frm_sum']

    to_process = []
    for h5_num in par['dat_images']:
        h5_start = 0
        h5_end = h5_num['num'] - par['frm_sum']
        h5_step = h5_num['num'] // par['frm_sum']
        assert h5_num['num'] % par['frm_sum'] == 0, 'Number of images must be an integer multiple of image summation!'
        h5_iter = np.linspace(h5_start, h5_end, h5_step, dtype=int)
        to_process.append((h5_num['link'], h5_iter))
    return to_process, par, progress_total

def convert_h5_to_cbf(h5_path, h5_idx, frm_num, par):
    import fabio.cbfimage as cbfimage
    frm_name = '{}_{:02}_{:04}.cbf'.format(os.path.join(par['frm_path'], par['frm_name']), par['frm_run'], frm_num)
    # Overwrite existing files if overwrite is flagged True
    if not par['overwrite'] and os.path.isfile(frm_name):
        return False
        
    # open the h5 file and sum the images in the given range
    with h5py.File(par['h5_file'], 'r') as h5f:
        frm_data = np.sum(h5f[h5_path][h5_idx:h5_idx+par['frm_sum'],:,:], axis=0)

    # set bad pixels to zero
    # this is only cosmetic but makes the indexing / visual analysis easier
    # as long as the xa mask files are used during integration
    # these pixels are masked and properly handled with
    frm_data[frm_data == par['det_max_counts'] * par['frm_sum']] = 0
    frm_data[frm_data < 0] = 0

    # scan parameters
    # increment, exposure time, start and end angle of the omega scan
    frm_idx = (frm_num - 1) * par['frm_sum']
    scn_inc = par['gon_scan_width'] * par['frm_sum']
    scn_exp = par['det_frame_time'] * par['frm_sum']
    scn_sta = par['gon_omega_start'][frm_idx]
    scn_end = par['gon_omega_end'][frm_idx + par['frm_sum'] - 1]

    # convert kappa to euler geometry
    omg_0, phi_0, chi_0 = kappa_to_euler(scn_sta, par['kappa'], par['alpha'], 0)
    omg_1, phi_1, chi_1 = kappa_to_euler(scn_end, par['kappa'], par['alpha'], 0)

    # angle correction
    phi_0 = 0.0 - phi_0
    phi_1 = 0.0 - phi_1
    omg_0 = omg_0 - 40.8
    omg_1 = omg_1 - 40.8

    cbfimage.cbfimage(data=frm_data).write(frm_name)
    #fabio.cbfimage.CBFImage(frm_name).header['wavelength'] = par['src_wavelength']
    #fabio.cbfimage.CBFImage(frm_name).header['distance'] = par['det_distance']
    #fabio.cbfimage.CBFImage(frm_name).header['exposure_time'] = par['det_frame_time']
    #fabio.cbfimage.CBFImage(frm_name).header['start_angle'] = omg_0
    #fabio.cbfimage.CBFImage(frm_name).header['end_angle'] = omg_1
    #fabio.cbfimage.CBFImage(frm_name).header['phi'] = phi_0
    #fabio.cbfimage.CBFImage(frm_name).header['chi'] = chi_0
    #fabio.cbfimage.CBFImage(frm_name).header['scan_increment'] = scn_inc
    #fabio.cbfimage.CBFImage(frm_name).header['scan_exposure'] = scn_exp
    #fabio.cbfimage.CBFImage(frm_name).header['scan_start'] = scn_sta
    #fabio.cbfimage.CBFImage(frm_name).header['scan_end'] = scn_end
    #fabio.cbfimage.CBFImage(frm_name).header['scan_number'] = frm_num
    #fabio.cbfimage.CBFImage(frm_name).header['scan_frames'] = par['frm_sum']
    #fabio.cbfimage.CBFImage(frm_name).header['scan_total'] = par['dat_images_num']
    #fabio.cbfimage.CBFImage(frm_name).header['scan_type'] = 'omega'
    #fabio.cbfimage.CBFImage(frm_name).header['scan_range'] = abs(scn_inc)
    #fabio.cbfimage.CBFImage(frm_name).header['scan_increment'] = scn_inc
    return True

def read_cbf_gz(fname):
    '''
     
    '''
    import gzip
    import re
    with gzip.open(fname, 'rb') as f:
        stream = f.read()
    start = stream.find(b'\x0c\x1a\x04\xd5') +4
    head = str(stream[:start])
    size = int(re.search(r'X-Binary-Size:\s+(\d+)', head).group(1))
    dim1 = int(re.search(r'X-Binary-Size-Fastest-Dimension:\s+(\d+)', head).group(1))
    dim2 = int(re.search(r'X-Binary-Size-Second-Dimension:\s+(\d+)', head).group(1))
    data = decByteOffset_np(stream[start:start+size]).reshape((dim2, dim1))
    return head, data

def decByteOffset_np(stream, dtype="int64"):
    '''
    The following code is taken from the FabIO package:
    Version: fabio-0.9.0
    Home-page: http://github.com/silx-kit/fabio
    Author: Henning Sorensen, Erik Knudsen, Jon Wright, Regis Perdreau,
            Jérôme Kieffer, Gael Goret, Brian Pauw, Valentin Valls
    '''
    """
    Analyze a stream of char with any length of exception:
                2, 4, or 8 bytes integers

    @param stream: string representing the compressed data
    @param size: the size of the output array (of longInts)
    @return: 1D-ndarray
    """
    import numpy as np
    listnpa = []
    key16 = b"\x80"
    key32 = b"\x00\x80"
    key64 = b"\x00\x00\x00\x80"
    shift = 1
    while True:
        idx = stream.find(key16)
        if idx == -1:
            listnpa.append(np.frombuffer(stream, dtype="int8"))
            break
        listnpa.append(np.frombuffer(stream[:idx], dtype="int8"))

        if stream[idx + 1:idx + 3] == key32:
            if stream[idx + 3:idx + 7] == key64:
                # 64 bits int
                res = np.frombuffer(stream[idx + 7:idx + 15], dtype="int64")
                listnpa.append(res)
                shift = 15
            else:
                # 32 bits int
                res = np.frombuffer(stream[idx + 3:idx + 7], dtype="int32")
                listnpa.append(res)
                shift = 7
        else:  # int16
            res = np.frombuffer(stream[idx + 1:idx + 3], dtype="int16")
            listnpa.append(res)
            shift = 3
        stream = stream[idx + shift:]
    return np.ascontiguousarray(np.hstack(listnpa), dtype).cumsum()

def get_run_info(basename):
    # try to get the run and frame number from the filename
    # any_name_runNum_frmNum is assumed.
    # maybe except for index error as well!
    try:
        _split = basename.split('_')
        frmTmp = _split.pop()
        frmLen = len(frmTmp)
        frmNum = int(frmTmp)
        runNum = int(_split.pop())
        stem = '_'.join(_split)
        return stem, runNum, frmNum, frmLen
    except ValueError:
        pass
    # SP-8 naming convention
    frmNum = int(basename[-3:])
    runNum = int(basename[-5:-3])
    stem = basename[:-6]
    return stem, runNum, frmNum, 3

def prepare_ESRF(path, *args):
    import glob
    fn_kwargs = {}
    imgs = sorted(glob.glob(os.path.join(path, '*.cbf.gz')))
    tail = ''
    while not tail:
        path, tail = os.path.split(path)
    sfrm_path = os.path.join(path, '_sfrm')
    if not os.path.exists(sfrm_path):
        os.mkdir(sfrm_path)
    progress_total = len(imgs)
    to_process = [(sfrm_path, imgs)]
    return to_process, fn_kwargs, progress_total

def convert_img_to_sfrm(sfrm_path, img_path, *args, **kwargs):
    '''
    
    '''
    import os, re
    import numpy as np
    from datetime import datetime as dt
    
    overwrite = kwargs.get('overwrite', False)
    rotate = kwargs.get('rotate', 0)

    # split path, name and extension
    cbf_ext, gz = os.path.splitext(img_path)
    no_ext, cbf = os.path.splitext(cbf_ext)
    basename = os.path.basename(no_ext)
    frame_stem, frame_run, frame_num, _ = get_run_info(basename)
    
    # output file format: some_name_rr_ffff.sfrm
    outName = os.path.join(sfrm_path, '{}_{:>02}_{:>04}.sfrm'.format(frame_stem, frame_run, frame_num))

    # check if file exists and overwrite flag
    if os.path.exists(outName) and not overwrite:
        return False
    
    # read in the frame
    header, data = read_cbf_gz(img_path)

    # rotate the data if needed
    data = np.rot90(data, rotate, axes=(1, 0))
    
    # the dead areas are flagged -1
    data[data == -1] = 0
    # bad pixels are flagged -2
    data[data == -2] = 0
    
    # scale the data to avoid underflow tables
    baseline_offset = -1 * data.min()
    data += baseline_offset
    
    # extract scan info from cbf header

    def regsearch(pattern, text, num=1, type=float):
        if match := re.search(pattern, text):
            return type(match.group(num))
        else:
            None

    sca_ext = regsearch(r'Exposure_time\s+(\d+\.\d+)\s+s', header, 1)
    sca_exp = regsearch(r'Exposure_period\s+(\d+\.\d+)\s+s', header, 1)
    gon_dxt = regsearch(r'Detector_distance\s+(\d+\.\d+)\s+m', header, 1) * 1000.0
    src_wav = regsearch(r'Wavelength\s+(\d+\.\d+)\s+A', header, 1)
    gon_phi = regsearch(r'Phi\s+(-*\d+\.\d+)\s+deg.', header, 1)
    inc_phi = regsearch(r'Phi_increment\s+(-*\d+\.\d+)\s+deg.', header, 1) or 0.0
    gon_chi = regsearch(r'Chi\s+(-*\d+\.\d+)\s+deg.', header, 1)
    inc_chi = regsearch(r'Chi_increment\s+(-*\d+\.\d+)\s+deg.', header, 1) or 0.0
    gon_kap = regsearch(r'Kappa\s+(-*\d+\.\d+)\s+deg.', header, 1)
    gon_alp = regsearch(r'Alpha\s+(-*\d+\.\d+)\s+deg.', header, 1) or 24.0
    gon_omg = regsearch(r'Omega\s+(-*\d+\.\d+)\s+deg.', header, 1)
    inc_omg = regsearch(r'Omega_increment\s+(-*\d+\.\d+)\s+deg.', header, 1) or 0.0
    gon_tth = regsearch(r'Detector_2theta\s+(-*\d+\.\d+)\s+deg.', header, 1)
    det_bcx = regsearch(r'Beam_xy\s+\((\d+\.\d+),\s+(\d+\.\d+)\)\s+pixels', header, 1)
    det_bcy = regsearch(r'Beam_xy\s+\((\d+\.\d+),\s+(\d+\.\d+)\)\s+pixels', header, 2)
    
    # initial frame dimensions are needed to calculate
    # the beamcenter of the reshaped frame and
    # adjust the beam center to the Bruker flip
    # numpy array starts in the upper left corner
    # Bruker starts lower left
    beam_y = data.shape[0] - det_bcy
    beam_x = det_bcx

    # convert kappa to euler
    if gon_kap is not None:
        gon_alp = 24.0
        gon_omg, gon_chi, gon_phi = kappa_to_euler(gon_omg, gon_kap, gon_alp, gon_phi)

    # to Bruker conversion:
    gon_tth = round(-gon_tth, 4)
    gon_omg = round(180.0 - gon_omg, 4)
    inc_omg = round(-inc_omg, 4)
    gon_chi = round(-gon_chi, 4)
    inc_chi = round(-inc_chi, 4)
    gon_phi = round(-gon_phi, 4)
    inc_phi = round(-inc_phi, 4)
    
    # ending positions
    end_phi = round(gon_phi + inc_phi, 4)
    end_chi = round(gon_chi + inc_chi, 4)
    end_omg = round(gon_omg + inc_omg, 4)
    end_tth = round(gon_tth, 4)
    
    # Scan axis (1=2-theta, 2=omega, 3=phi, 4=chi)
    ang_nam = ['Omega',  'Phi']
    ang_inc = [inc_omg, inc_phi]
    ang_sta = [gon_omg, gon_phi]
    sca_nam, sca_axs, sca_sta, sca_inc = [(ang_nam[i], int(i+2), ang_sta[i], round(v,4)) for i,v in enumerate(ang_inc) if v != 0.0][0]
    
    # calculate detector pixel per cm
    # this is normalized to a 512x512 detector format
    # EIGER2 pixel size is 0.075 mm 
    pix_per_512 = round((10.0 / 0.075) * (512.0 / data.shape[1]), 6)
    
    # default bruker header
    header = init_bruker_header()
    
    # fill known header items
    header['NCOLS']      = data.shape[1]                             # Number of pixels per row; number of mosaic tiles in X; dZ/dX
    header['NROWS']      = data.shape[0]                             # Number of rows in frame; number of mosaic tiles in Y; dZ/dY value
    header['CENTER']     = [beam_x, beam_y, beam_x, beam_y]          # 
    header['CCDPARM']    = [0.00, 1.00, 1.00, 1.00, 1169523]         # CCD parameters for computing pixel ESDs; readnoise, e/ADU, e/photon, bias, full scale
    header['DETPAR']     = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]          # Detector position corrections (Xc,Yc,Dist,Pitch,Roll,Yaw)
    header['CHEM']       = '?'                                                   
    header['DETTYPE']    = ['EIGER2-16M', pix_per_512, 0.00, 0, 0.001, 0.0, 0]
    header['SITE']       = 'ESRF/I19-1'                              # Site name
    header['MODEL']      = 'Synchrotron'                             # Diffractometer model
    header['TARGET']     = 'Undulator'                               # X-ray target material)
    header['USER']       = '?'                                       # Username
    header['SOURCEK']    = '?'                                       # X-ray source kV
    header['SOURCEM']    = '?'                                       # Source milliamps
    header['WAVELEN']    = [src_wav, src_wav, src_wav]               # Wavelengths (average, a1, a2)
    header['FILENAM']    = basename
    header['CUMULAT']    = sca_exp                                   # Accumulated exposure time in real hours
    header['ELAPSDR']    = sca_ext                                   # Requested time for this frame in seconds
    header['ELAPSDA']    = sca_exp                                   # Actual time for this frame in seconds
    header['START']      = sca_sta                                   # Starting scan angle value, decimal deg
    header['ANGLES']     = [gon_tth, gon_omg, gon_phi, gon_chi]      # Diffractometer setting angles, deg. (2Th, omg, phi, chi)
    header['ENDING']     = [end_tth, end_omg, end_phi, end_chi]      # Setting angles read at end of scan
    header['TYPE']       = 'Generic {} Scan'.format(sca_nam)         # String indicating kind of data in the frame
    header['DISTANC']    = float(gon_dxt) / 10.0                     # Sample-detector distance, cm
    header['RANGE']      = abs(sca_inc)                              # Magnitude of scan range in decimal degrees
    header['INCREME']    = sca_inc                                   # Signed scan angle increment between frames
    header['NUMBER']     = frame_num                                 # Number of this frame in series (zero-based)
    header['NFRAMES']    = '?'                                       # Number of frames in the series
    header['AXIS']       = sca_axs                                   # Scan axis (1=2-theta, 2=omega, 3=phi, 4=chi)
    header['LOWTEMP']    = [1, 0, 0]                                 # Low temp flag; experiment temperature*100; detector temp*100
    header['NEXP']       = [1, 0, baseline_offset, 0, 0]
    header['MAXXY']      = np.array(np.where(data == data.max()), float)[:, 0]
    header['MAXIMUM']    = np.max(data)
    header['MINIMUM']    = np.min(data)
    header['NCOUNTS']    = [data.sum(), 0]
    header['NPIXELB']    = [1, 1]                                    # bytes/pixel in main image, bytes/pixel in underflow table
    header['NOVER64']    = [data[data > 64000].shape[0], 0, 0]
    header['NSTEPS']     = 1                                         # steps or oscillations in this frame
    header['COMPRES']    = 'NONE'                                    # compression scheme if any
    header['TRAILER']    = 0                                         # byte pointer to trailer info
    header['LINEAR']     = [1.00, 0.00]     
    header['PHD']        = [1.00, 0.00]
    header['OCTMASK']    = [0, 0, 0, 1023, 1023, 2046, 1023, 1023]
    header['DISPLIM']    = [0.0, 63.0]                               # Recommended display contrast window settings
    header['FILTER2']    = [90.0, 0.0, 0.0, 1.0]                     # Monochromator 2-theta, roll (both deg)
    #header['CREATED']    = [dt.fromtimestamp(os.path.getmtime(img_path)).strftime('%Y-%m-%d %H:%M:%S')]# use creation time of raw data!
    
    # write the frame
    write_bruker_frame(outName, header, data)
    return True
