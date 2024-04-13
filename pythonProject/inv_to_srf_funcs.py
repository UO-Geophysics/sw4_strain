def fault_segments(file_path, called_from_mud2srf=False):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    cdict = {'red': ((0., 1, 1),
                     (0.03, 1, 1),
                     (0.20, 0, 0),
                     (0.66, 1, 1),
                     (0.89, 1, 1),
                     (1, 0.5, 0.5)),
             'green': ((0., 1, 1),
                       (0.03, 1, 1),
                       (0.20, 0, 0),
                       (0.375, 1, 1),
                       (0.64, 1, 1),
                       (0.91, 0, 0),
                       (1, 0, 0)),
             'blue': ((0., 1, 1),
                      (0.08, 1, 1),
                      (0.20, 1, 1),
                      (0.34, 1, 1),
                      (0.65, 0, 0),
                      (1, 0, 0))}

    whitejet = matplotlib.colors.LinearSegmentedColormap('whitejet', cdict, 256)

    ### Separate the first section of the file with the earliest onset times ###

    f = np.genfromtxt(file_path)  # Read mudpy file
    first_section_first_subfault_num = f[0, 0]  # Where does the first section start
    section_start_idxs = np.where(f[:, 0] == first_section_first_subfault_num)[
        0]  # Find where the subfault numbers start over
    first_section_last_subfault_num = f[:, 0][
        section_start_idxs[1] - 1]  # Last subfault number is one before the first restart
    # print(first_section_last_subfault_num)

    ### Sum the slips across ALL sections ###

    all_num = f[:, 0]
    all_ss = f[:, 8]
    all_ds = f[:, 9]
    unum = np.unique(all_num)
    ss = np.zeros(len(unum))
    ds = np.zeros(len(unum))
    for k in range(len(unum)):
        i = np.where(unum[k] == all_num)
        ss[k] = all_ss[i].sum()
        ds[k] = all_ds[i].sum()
    # Sum them
    ss_slip = ss
    ds_slip = ds
    slip = (ss ** 2 + ds ** 2) ** 0.5

    ### Separate the segments ###

    section_start_row = section_start_idxs[0]
    section_end_row = section_start_idxs[1] - 1  # Last row/subfault of the section

    # Visualize segments of the section

    # Get other data just from the first section, since we already dealt with the total slip
    lon = f[section_start_row:section_end_row + 1, 1]  # +1 because slicing excludes the final number
    lat = f[section_start_row:section_end_row + 1, 2]
    depth = -f[section_start_row:section_end_row + 1, 3]
    strike = f[section_start_row:section_end_row + 1, 4]
    onset = f[section_start_row:section_end_row + 1, 12]

    # Separate indices for each of the four fault segments

    f4 = np.where((lon > -117.6) & (lon < -117.475) & (lat < 35.775) & (lat > 35.675) & (strike == 309.45))[0]

    f5 = np.where((lon > -117.7) & (lon < -117.62) & (lat < 35.84) & (lat > 35.79) & (strike == 309.84))[0]

    pref6 = np.where((lon > -117.5) & (lon < -117.38) & (lat < 35.675) & (lat > 35.55) & (strike > 311))[0]

    x1 = lon[pref6][0]  # This one is harder to separate, made a line and kept points below the line
    x2 = lon[pref6][10]
    y1 = lat[pref6][0]
    y2 = lat[pref6][10]
    m = (y2 - y1) / (x2 - x1)
    b = y2 - m * x2 - 0.01
    x = np.arange(-117.6, -117.3, 0.005)
    y = m * x + b

    belowline = []
    for i in range(len(lon[pref6])):
        truex = lon[pref6][i]
        ytest = m * truex + b
        truey = lat[pref6][i]
        if truey <= ytest:
            belowline.append(i)

    f6 = pref6[belowline]

    all_lon_idxs = np.arange(0, len(lon), 1)
    others = np.append(f4, f5)
    others = np.append(others, f6)

    f3 = np.delete(all_lon_idxs, others)

    segments = [f3, f4, f5, f6]

    if called_from_mud2srf == False:  # Make the plots

        # Plot the segments in separate colors - map view

        plt.title(file_path)
        plt.scatter(lon[f3], lat[f3], s=4, color='blue', label='F3')
        plt.scatter(lon[f4], lat[f4], s=4, color='red', label='F4')
        plt.scatter(lon[f5], lat[f5], s=4, color='orange', label='F5')
        plt.scatter(lon[f6], lat[f6], s=4, color='turquoise', label='F6')
        plt.legend()

        plt.show()

        # Plot them 3D view

        marker_size = 10
        clims = None
        plot_onset = False
        cmap = whitejet

        if plot_onset == False:
            plot_variable = slip
        else:
            plot_variable = onset

        fig = plt.figure(figsize=(14, 4))
        ax = fig.add_subplot(111, projection='3d')

        if clims == None:
            p = ax.scatter(lon, lat, depth, c=plot_variable, cmap=cmap, marker='o', s=marker_size, lw=0)
            # f3 = ax.scatter(lon[f3], lat[f3], depth[f3], c = plot_variable[f3], cmap = cmap, marker = 'o', s = marker_size, lw = 0)
            # f4 = ax.scatter(lon[f4], lat[f4], depth[f4], c = plot_variable[f4], cmap = cmap, marker = 's', s = marker_size, lw = 0)
            # f5 = ax.scatter(lon[f5], lat[f5], depth[f5], c = plot_variable[f5], cmap = cmap, marker = 's', s = marker_size, lw = 0)
            # f6 = ax.scatter(lon[f6], lat[f6], depth[f6], c = plot_variable[f6], cmap = cmap, marker = 's', s = marker_size, lw = 0)
        else:
            p = ax.scatter(lon, lat, depth, c=plot_variable, cmap=cmap, marker='o', s=marker_size, vmin=clims[0],
                           vmax=clims[1], lw=0)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Depth (km)')
        cb = fig.colorbar(p)

        if plot_onset == False:
            cb.set_label('Slip (m)')
        else:
            cb.set_label('Onset time (s)')

        plt.subplots_adjust(left=0.1, bottom=0.1, right=1.0, top=0.9, wspace=0, hspace=0)

        plt.show()

    return segments


def write_segment_headers(file_path, inv_or_rupt, fout, segment_plots):
    import numpy as np
    from os import path
    import matplotlib.pyplot as plt
    from obspy.geodetics import degrees2kilometers, gps2dist_azimuth

    # Define paths to output files
    folder = path.dirname(file_path)
    basename = path.basename(file_path)

    if inv_or_rupt == 'rupt':
        basename = basename.replace('rupt', 'srf')
        srf_file = folder + '/' + basename
    else:
        basename = basename.replace('inv', 'srf')
        srf_file = folder + '/' + basename

        ### Figure out where the inversion windows are from subfault number repeats ###

    f = np.genfromtxt(file_path)  # Read mudpy file
    first_section_first_subfault_num = f[0, 0]  # Where does the first section start
    section_start_idxs = np.where(f[:, 0] == first_section_first_subfault_num)[
        0]  # Find where the subfault numbers start over
    first_section_last_subfault_num = f[:, 0][
        section_start_idxs[1] - 1]  # Last subfault number is one before the first restart

    sec_start = section_start_idxs[0]
    sec_end = section_start_idxs[1] - 1

    ### Extract data from first section for the segment header ###

    num = f[sec_start:sec_end + 1, 0]  # +1 because slicing excludes the final number
    lon = f[sec_start:sec_end + 1, 1]
    lat = f[sec_start:sec_end + 1, 2]
    depth = f[sec_start:sec_end + 1, 3]  # km
    strike = f[sec_start:sec_end + 1, 4]
    dip = f[sec_start:sec_end + 1, 5]
    ss_len = f[sec_start:sec_end + 1, 10]  # m
    ds_len = f[sec_start:sec_end + 1, 11]  # m
    onset = f[sec_start:sec_end + 1, 12]  # rupt_time(s)

    # print(num.shape)
    # print(lon.shape)
    # print(lat.shape)
    # print(depth.shape)
    # print(strike.shape)
    # print(dip.shape)
    # print(ss_len.shape)
    # print(ds_len.shape)
    # print(onset.shape)

    print('Writing segments info')

    fout.write('2.0\n')  # SRF version

    segments = fault_segments(file_path, called_from_mud2srf=True)
    # test_seg = segments[0]
    counter = 1

    fout.write('PLANE %d\n' % (len(segments)))

    ### Start of segment loop

    for seg in segments:

        plane_num = counter
        print('')
        print('Processing segment ' + str(plane_num))
        counter += 1

        num_seg = num[seg]
        lon_seg = lon[seg]
        lat_seg = lat[seg]
        depth_seg = depth[seg]  # km
        strike_seg = strike[seg]
        dip_seg = dip[seg]
        ss_len_seg = ss_len[seg]  # m
        ds_len_seg = ds_len[seg]  # m
        onset_seg = onset[seg]  # rupt_time(s)

        # print(num_seg.shape)
        # print(lon_seg.shape)
        # print(lat_seg.shape)
        # print(depth_seg.shape)
        # print(strike_seg.shape)
        # print(dip_seg.shape)
        # print(ss_len_seg.shape)
        # print(ds_len_seg.shape)
        # print(onset_seg.shape)

        # Get coordinates of shallowest row
        tol = 0.5
        min_row = np.where(np.abs(depth_seg - min(depth_seg)) < tol)[0]  # Minimum depth

        # ELON: Top center longitude
        elon = np.mean(lon_seg[min_row])

        # ELAT: Top center latitude
        elat = np.mean(lat_seg[min_row])

        # NSTK & NDIP: number of subfaults along strike and dip (calculated from shallowest row)
        Nstrike = len(num_seg[min_row])
        Ndip = len(num_seg) / Nstrike

        print('Segment total number of subfaults: ' + str(len(num_seg)))
        print('Segment total number of subfaults (with 5 windows): ' + str(len(num_seg) * 5))
        print('Segment subfault dimensions: ' + str(Nstrike) + ' x ' + str(round(Ndip, 2)))

        if segment_plots:
            # Depth plot sanity check
            plt.title('F' + str(plane_num + 2))
            plt.scatter(lon_seg, -depth_seg, label='All subfaults')  # All lons, all depths
            plt.scatter(lon_seg[min_row], -depth_seg[min_row],
                        label='Shallowest row')  # Shallowest row with 500m tolerance
            plt.scatter(elon, -np.mean(depth_seg[min_row]), label='Top center of segment')  # Top center longitude
            plt.ylabel('Depth (km)')
            plt.xlabel('Longitude')
            plt.legend(loc='lower right')
            plt.show()

        # LEN & WID: Segment length and width

        # Length
        min_lon = min(lon_seg)
        max_lon = max(lon_seg)
        a = np.where(lon_seg == min_lon)[0]
        min_lon = lon_seg[a[0]]
        min_lat = lat_seg[a[0]]
        b = np.where(lon_seg == max_lon)[0]
        max_lon = lon_seg[b[0]]
        max_lat = lat_seg[b[0]]

        if segment_plots:
            # Verify you have the right endpoints for the fault segment
            plt.title('F' + str(plane_num + 2))
            plt.scatter(lon_seg, lat_seg, label='Entire segment', color='blue')
            plt.scatter(min_lon, min_lat, label='Min endpoint', color='orange')
            plt.scatter(max_lon, max_lat, label='Max endpoint', color='lime')

        segment_len = gps2dist_azimuth(min_lat, min_lon, max_lat, max_lon)[0] / 1000
        
        print(min_lon, max_lon, min_lat, max_lat)

        # Width

        min_depth = min(depth_seg)
        max_depth = max(depth_seg)

        segment_width = max_depth - min_depth

        # STK & DIP: Segment strike and dip

        segment_strike = np.mean(strike_seg)
        segment_dip = np.mean(dip_seg)

        # DTOP: Depth to top of fault

        # Average subfault in shallowest row of segment
        subfault_ss_len = np.mean(ss_len_seg[min_row]) / 1000  # km
        subfault_ds_len = np.mean(ds_len_seg[min_row]) / 1000  # km

        depth_to_shallowest_subfault = np.mean(depth_seg[min_row])
        shallowest_subfault_dip = np.mean(dip_seg[min_row])
        depth_to_top = np.abs(
            depth_to_shallowest_subfault - np.sin(np.deg2rad(shallowest_subfault_dip)) * (subfault_ds_len / 2))
        # See notebook drawing for geometry

        # Next need to find the hypocenter of this fault segment - look for where the lowest rupt_time(s) is (rupture start time)
        earliest_rupt_start_time = min(onset_seg)
        m = np.where(onset_seg == earliest_rupt_start_time)[0]
        hypo_lon = lon_seg[m[0]]
        hypo_lat = lat_seg[m[0]]
        hypo_depth = depth_seg[m[0]]

        # SYHP: along-strike location from top center of hypo for this segment (km)

        straight_segment_center_lon = (max_lon + min_lon) / 2
        straight_segment_center_lat = (max_lat + min_lat) / 2

        # Find closest point on curving fault
        dists = []
        sublons = []
        sublats = []
        for idx2 in range(len(num_seg)):
            sublon = lon_seg[idx2]
            sublat = lat_seg[idx2]
            dist = np.sqrt((straight_segment_center_lon - sublon) ** 2 + (straight_segment_center_lat - sublat) ** 2)
            dists.append(dist)
            sublons.append(sublon)
            sublats.append(sublat)

        mindist = min(dists)
        n = np.where(dists == mindist)[0]
        center_on_fault_lon = sublons[n[0]]
        center_on_fault_lat = sublats[n[0]]

        if segment_plots:
            # Add to sanity check plot
            plt.scatter(hypo_lon, hypo_lat, label='Hypocenter', color='red')
            plt.scatter(straight_segment_center_lon, straight_segment_center_lat, label='Straight segment center',
                        color='purple')
            plt.scatter(center_on_fault_lon, center_on_fault_lat, label='On-fault segment center', color='turquoise')
            plt.legend(loc='upper right')
            plt.show()

        which_center = 'straight'  # or 'on-fault'

        if which_center == 'straight':
            center_lon = straight_segment_center_lon
            center_lat = straight_segment_center_lat
        else:
            center_lon = center_on_fault_lon
            center_lat = center_on_fault_lat

        print('Using ' + which_center + ' segment center for SHYP distance')

        prelim_shyp, az, baz = gps2dist_azimuth(hypo_lat, hypo_lon, center_lat, center_lon)  # shyp in m

        # Should shyp be positive or negative?
        if segment_strike > 180:
            segment_strike_rectified = segment_strike - 360
        else:
            segment_strike_rectified = segment_strike

        if np.sign(az) == np.sign(segment_strike_rectified):
            shyp = -prelim_shyp / 1000  # km
        else:
            shyp = prelim_shyp / 1000  # km

        # print(shyp)

        # DHYP: along-dip location from top edge of hypo for this segment (km)

        # See notebook drawings for geometry
        dhyp = (hypo_depth - depth_to_top) / np.sin(np.deg2rad(segment_dip))
        # print(dhyp)

        # FULL SEGMENT HEADER

        # Write header data
        fout.write('  %.6f\t%.6f\t%d\t%d\t%.4f\t%.4f\n' % (elon, elat, Nstrike, Ndip, segment_len, segment_width))
        fout.write('  %.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % (segment_strike, segment_dip, depth_to_top, shyp, dhyp))

    return section_start_idxs, segments


def write_point_stf(fout, slip_kfault, total_time_kfault, stf_dt, rise_time_kfault, stf_type, time_pad, tinit_kfault,
                    minSTFpoints, minNTstf, lon_kfault, lat_kfault, depth_kfault, strike_kfault, dip_kfault,
                    area_kfault, vs_kfault, density_kfault, rake_kfault, zero_slip, integrate):
    import numpy as np
    from mudpy import forward
    from scipy.integrate import cumtrapz

    if slip_kfault == 0:
        zero_slip = True
        stf = np.zeros(int(total_time_kfault / stf_dt))
        # print('Zero slip at ' + str(kfault))
    else:
        tstf, stf = forward.build_source_time_function(rise_time_kfault, stf_dt, total_time_kfault, stf_type=stf_type,
                                                       zeta=0.2, scale=True)
        # Scale stf so integral matches total slip
        stf_adjust_factor = slip_kfault / stf_dt
        stf = stf * stf_adjust_factor  # now tf is in cm/sec
    # Now zero pad before and after end because SW4 struggles if subfault STFs are not zero padded
    if time_pad != None:
        zeros_pad = np.zeros(int(time_pad / stf_dt))
        stf = np.r_[zeros_pad, stf, zeros_pad]
        # Change start time of STF, it should now begin time_pad seconds earlier
        tinit_kfault = tinit_kfault - time_pad
    # How many STF points?
    NTstf = len(stf)
    if NTstf < minSTFpoints:  # Too short, zero pad
        print('Padding short STF...')
        zeros_pad = np.zeros(int(minSTFpoints / 2))
        stf = np.r_[zeros_pad, stf, zeros_pad]
        # Change start time of STF, it should now begin time_pad seconds earlier
        time_shift = int(minSTFpoints / 2) * stf_dt
        tinit_kfault = tinit_kfault - time_shift
        # Check that everything is ok
    NTstf = len(stf)
    if NTstf < minNTstf:
        minNTstf = NTstf
    if zero_slip == True:
        NTstf = 0

    if zero_slip == False:

        # Integrate to slip instead of slip rate?
        if integrate == True:
            t = np.arange(0, len(stf) * stf_dt, stf_dt)
            if len(t) > len(stf):
                t = t[0:-1]
            # print(t.shape)
            # print(stf.shape)
            stf_integrated = cumtrapz(stf, t, initial=0)
            stf = stf_integrated

        # Subfault header
        fout.write('  %.6f  %.6f  %.5e  %.2f  %.2f  %.5e  %.4f  %.4e  %.4e  %.4e\n' % (
        lon_kfault, lat_kfault, depth_kfault, strike_kfault, dip_kfault, area_kfault, tinit_kfault, stf_dt, vs_kfault,
        density_kfault))
        fout.write('  %.2f  %.4f  %d  0  0  0  0\n' % (rake_kfault, slip_kfault, NTstf))

        # Write stf 6 values per line
        for kstf in range(NTstf):
            if kstf == 0:
                white_space = '  '
            elif (kstf + 1) % 6 == 0:
                white_space = '\n'
            elif (kstf + 1) == NTstf:
                white_space = '\n'
            else:
                white_space = '  '
            if kstf == 0:
                pre_white_space = '  '
            elif (kstf) % 6 == 0:
                pre_white_space = '  '
            else:
                pre_white_space = ''

            fout.write('%s%.6e%s' % (pre_white_space, stf[kstf], white_space))


def write_srf(file_path, inv_or_rupt, segment_plots, stf_dt=0.1, stf_type='triangle',
              time_pad=5.0, minSTFpoints=16, integrate=False):
    '''
    Convert a mudpy .rupt or .inv file to SRF version 2 format
    Works with faults with multiple segments - this one is set up specifically for the M7
    Ridgecrest mainshock using Dara Goldberg's .inv file

    See for SRF format description here:
        https://scec.usc.edu/scecpedia/Standard_Rupture_Format

    Function modified from mudpy forward.mudpy2srf by Sydney Dybing in April 2024

    file_path = path to .inv or .rupt file being used
    inv_or_rupt can be = 'inv' or 'rupt'
    segment_plots can be True or False to output plots of the fault segments
    '''

    import numpy as np
    from os import path

    # Define paths to output files
    folder = path.dirname(file_path)
    basename = path.basename(file_path)

    if inv_or_rupt == 'rupt':
        basename = basename.replace('rupt', 'srf')
        srf_file = folder + '/' + basename
    else:
        basename = basename.replace('inv', 'srf')
        srf_file = folder + '/' + basename

        # Open SRF file
    fout = open(srf_file, 'w')

    # Write segment headers and get the section start indices
    section_start_idxs, segments = write_segment_headers(file_path, inv_or_rupt, fout, segment_plots)

    # Load the mudpy file
    f = np.genfromtxt(file_path)  # Read mudpy file

    # Loop through segments to write the points section of the SRF

    counter = 1

    for seg in segments:

        plane_num = counter
        print('')
        print('Getting data for segment ' + str(plane_num))
        counter += 1

        # Get section start and end indices

        sec1_start = section_start_idxs[0]
        sec1_end = section_start_idxs[1] - 1
        sec2_start = section_start_idxs[1]
        sec2_end = section_start_idxs[2] - 1
        sec3_start = section_start_idxs[2]
        sec3_end = section_start_idxs[3] - 1
        sec4_start = section_start_idxs[3]
        sec4_end = section_start_idxs[4] - 1
        sec5_start = section_start_idxs[4]
        sec5_end = int(len(f))

        # Data that doesn't change with section - just get it all from first section

        num_seg, lon_seg, lat_seg, depth_seg, strike_seg, dip_seg, rise_seg, dura_seg, ss_len_seg, ds_len_seg, rigidity_seg = unchanging_data(
            f, sec1_start, sec1_end, seg)

        # Data that does change with section

        ss_slip1_seg, ds_slip1_seg, slip1_seg, onset1_seg = section_data(f, sec1_start, sec1_end, seg)
        ss_slip2_seg, ds_slip2_seg, slip2_seg, onset2_seg = section_data(f, sec2_start, sec2_end, seg)
        ss_slip3_seg, ds_slip3_seg, slip3_seg, onset3_seg = section_data(f, sec3_start, sec3_end, seg)
        ss_slip4_seg, ds_slip4_seg, slip4_seg, onset4_seg = section_data(f, sec4_start, sec4_end, seg)
        ss_slip5_seg, ds_slip5_seg, slip5_seg, onset5_seg = section_data(f, sec5_start, sec5_end, seg)

        print('Processing points')

        # Get coordinates of shallowest row
        tol = 0.5
        min_row = np.where(np.abs(depth_seg - min(depth_seg)) < tol)[0]  # Minimum depth
        Nstrike = len(num_seg[min_row])
        Ndip = len(num_seg) / Nstrike

        fout.write('POINTS %d\n' % (Nstrike * Ndip * 5))

        # Get the subfault source time functions
        # Note mudpy works in mks; SRF is cgs so must convert accordingly

        minNTstf = 99999

        for kfault in range(len(num_seg)):
            # print(kfault)
            zero_slip = False

            # Get values for "Headers" for this subfault
            # Data that doesn't change with window
            lon_kfault = lon_seg[kfault]
            lat_kfault = lat_seg[kfault]
            depth_kfault = depth_seg[kfault]
            strike_kfault = strike_seg[kfault]
            dip_kfault = dip_seg[kfault]
            area_kfault = ss_len_seg[kfault] * ds_len_seg[kfault] * 100 ** 2  # ss_len(m) * ds_len(m) * 100**2 in cm^2

            # Slips and other info from each window for the segment
            ss_slip1_kfault, ds_slip1_kfault, rake1_kfault, slip1_kfault, tinit1_kfault = subfault_section_data(kfault,
                                                                                                                ss_slip1_seg,
                                                                                                                ds_slip1_seg,
                                                                                                                onset1_seg)
            ss_slip2_kfault, ds_slip2_kfault, rake2_kfault, slip2_kfault, tinit2_kfault = subfault_section_data(kfault,
                                                                                                                ss_slip2_seg,
                                                                                                                ds_slip2_seg,
                                                                                                                onset2_seg)
            ss_slip3_kfault, ds_slip3_kfault, rake3_kfault, slip3_kfault, tinit3_kfault = subfault_section_data(kfault,
                                                                                                                ss_slip3_seg,
                                                                                                                ds_slip3_seg,
                                                                                                                onset3_seg)
            ss_slip4_kfault, ds_slip4_kfault, rake4_kfault, slip4_kfault, tinit4_kfault = subfault_section_data(kfault,
                                                                                                                ss_slip4_seg,
                                                                                                                ds_slip4_seg,
                                                                                                                onset4_seg)
            ss_slip5_kfault, ds_slip5_kfault, rake5_kfault, slip5_kfault, tinit5_kfault = subfault_section_data(kfault,
                                                                                                                ss_slip5_seg,
                                                                                                                ds_slip5_seg,
                                                                                                                onset5_seg)

            ##### EDIT TO DRAW INFO FROM VELOCITY MOD?
            vs_kfault = 2.80000e+05  # default value for not known
            density_kfault = 2.70000e+00  # default value for not known

            # Now get source time function
            rise_time_kfault = dura_seg[kfault]
            total_time_kfault = rise_time_kfault * 1.5  # Just pads the stf, doesn't actually moment

            ### Do for each window ###

            write_point_stf(fout, slip1_kfault, total_time_kfault, stf_dt, rise_time_kfault, stf_type, time_pad,
                            tinit1_kfault, minSTFpoints, minNTstf, lon_kfault, lat_kfault, depth_kfault, strike_kfault,
                            dip_kfault, area_kfault, vs_kfault, density_kfault, rake1_kfault, zero_slip, integrate)
            write_point_stf(fout, slip2_kfault, total_time_kfault, stf_dt, rise_time_kfault, stf_type, time_pad,
                            tinit2_kfault, minSTFpoints, minNTstf, lon_kfault, lat_kfault, depth_kfault, strike_kfault,
                            dip_kfault, area_kfault, vs_kfault, density_kfault, rake2_kfault, zero_slip, integrate)
            write_point_stf(fout, slip3_kfault, total_time_kfault, stf_dt, rise_time_kfault, stf_type, time_pad,
                            tinit3_kfault, minSTFpoints, minNTstf, lon_kfault, lat_kfault, depth_kfault, strike_kfault,
                            dip_kfault, area_kfault, vs_kfault, density_kfault, rake3_kfault, zero_slip, integrate)
            write_point_stf(fout, slip4_kfault, total_time_kfault, stf_dt, rise_time_kfault, stf_type, time_pad,
                            tinit4_kfault, minSTFpoints, minNTstf, lon_kfault, lat_kfault, depth_kfault, strike_kfault,
                            dip_kfault, area_kfault, vs_kfault, density_kfault, rake4_kfault, zero_slip, integrate)
            write_point_stf(fout, slip5_kfault, total_time_kfault, stf_dt, rise_time_kfault, stf_type, time_pad,
                            tinit5_kfault, minSTFpoints, minNTstf, lon_kfault, lat_kfault, depth_kfault, strike_kfault,
                            dip_kfault, area_kfault, vs_kfault, density_kfault, rake5_kfault, zero_slip, integrate)

    fout.close()

    print('')
    print('Done!')


def unchanging_data(f, sec1_start, sec1_end, seg, print_shapes=False):
    num_seg = f[sec1_start:sec1_end + 1, 0][seg]  # +1 because slicing excludes the final number
    lon_seg = f[sec1_start:sec1_end + 1, 1][seg]
    lat_seg = f[sec1_start:sec1_end + 1, 2][seg]
    depth_seg = f[sec1_start:sec1_end + 1, 3][seg]  # km
    strike_seg = f[sec1_start:sec1_end + 1, 4][seg]
    dip_seg = f[sec1_start:sec1_end + 1, 5][seg]
    rise_seg = f[sec1_start:sec1_end + 1, 6][seg]
    dura_seg = f[sec1_start:sec1_end + 1, 7][seg]
    ss_len_seg = f[sec1_start:sec1_end + 1, 10][seg]  # m
    ds_len_seg = f[sec1_start:sec1_end + 1, 11][seg]  # m
    rigidity_seg = f[sec1_start:sec1_end + 1, 13][seg]  # Pa

    if print_shapes:
        print(num_seg.shape)
        print(lon_seg.shape)
        print(lat_seg.shape)
        print(depth_seg.shape)
        print(strike_seg.shape)
        print(dip_seg.shape)
        print(rise_seg.shape)
        print(dura_seg.shape)
        print(ss_len_seg.shape)
        print(ds_len_seg.shape)
        print(rigidity_seg.shape)

    return num_seg, lon_seg, lat_seg, depth_seg, strike_seg, dip_seg, rise_seg, dura_seg, ss_len_seg, ds_len_seg, rigidity_seg


def section_data(f, sec_start, sec_end, seg, print_shapes=False):
    ss_slip_seg = f[sec_start:sec_end + 1, 8][seg]
    ds_slip_seg = f[sec_start:sec_end + 1, 9][seg]
    slip_seg = (ss_slip_seg ** 2 + ds_slip_seg ** 2) ** 0.5  # m
    onset_seg = f[sec_start:sec_end + 1, 12][seg]  # rupt_time(s)

    if print_shapes:
        print(ss_slip_seg.shape)
        print(ds_slip_seg.shape)
        print(slip_seg.shape)
        print(onset_seg.shape)

    return ss_slip_seg, ds_slip_seg, slip_seg, onset_seg


def subfault_section_data(kfault, ss_slip_seg, ds_slip_seg, onset_seg):
    import numpy as np

    ss_slip_kfault = ss_slip_seg[kfault]
    ds_slip_kfault = ds_slip_seg[kfault]
    rake_kfault = np.rad2deg(np.arctan2(ds_slip_kfault, ss_slip_kfault))  # ds_slip, ss_slip
    slip_kfault = np.sqrt(ss_slip_kfault ** 2 + ds_slip_kfault ** 2) * 100  # cm
    tinit_kfault = onset_seg[kfault]  # rupt_time(s)

    return ss_slip_kfault, ds_slip_kfault, rake_kfault, slip_kfault, tinit_kfault