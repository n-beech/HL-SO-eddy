&diag_list
ldiag_solver     =.false.
lcurt_stress_surf=.true.
ldiag_curl_vel3  =.false.
ldiag_energy     =.false.
ldiag_salt3D     =.false.
ldiag_dMOC       =.false.
ldiag_DVD        =.false.
ldiag_forc       =.false.
/

&nml_general
io_listsize=100 !number of streams to allocate. shallbe large or equal to the number of streams in &nml_list
/

! for sea ice related variables use_ice should be true, otherewise there will be no output
! for 'curl_surf' to work lcurt_stress_surf must be .true. otherwise no output
! for 'fer_C', 'bolus_u', 'bolus_v', 'bolus_w', 'fer_K' to work Fer_GM must be .true. otherwise no output
! 'otracers' - all other tracers if applicable
! for 'dMOC' to work ldiag_dMOC must be .true. otherwise no output
&nml_list
io_list =  'sst       ',1  , 'd', 4,
           'sss       ',1  , 'd', 4,
    	   'ssh       ',1  , 'd', 4,
           'uice      ',1, 'd', 4,
           'vice      ',1, 'd', 4,
           'a_ice     ',1, 'd', 4,
           'm_ice     ',1, 'd', 4,
           'm_snow    ',1, 'd', 4,
           'MLD1      ',1, 'd', 4,
           'MLD2      ',1, 'd', 4,
           'tx_sur    ',1, 'd', 4,
           'ty_sur    ',1, 'd', 4,
           'temp      ',1, 'd', 4,
           'salt      ',1, 'd', 8,
           !'N2        ',1, 'y', 4,
           !'Kv        ',1, 'y', 4,
           'unod         ',1, 'd', 4,
           'vnod         ',1, 'd', 4,
           'w         ',1, 'd', 4,
           !'Av        ',1, 'y', 4,
           !'bolus_u   ',1, 'y', 4,
           !'bolus_v   ',1, 'y', 4,
           !'bolus_w   ',1, 'y', 4,
	   'fh        ',1, 'd', 4,
	   'dMOC      ',1, 'm', 4,
/
