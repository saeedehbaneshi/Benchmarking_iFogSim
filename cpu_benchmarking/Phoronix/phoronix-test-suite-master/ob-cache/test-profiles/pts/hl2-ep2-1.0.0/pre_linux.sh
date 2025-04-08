#!/bin/bash

STEAM_GAME_ID=420
GAME_BINARY=hl2_linux

export HOME=$DEBUG_REAL_HOME

steam steam://run/$STEAM_GAME_ID -applaunch 420 $@ &
sleep 12
GAME_PID=`pgrep $GAME_BINARY`
echo '#!/bin/sh' > steam-env-vars.sh
while read -d $'\0' ENV; do NAME=`echo $ENV | cut -d= -f1`; VAL=`echo $ENV | cut -d= -f2`; echo "export $NAME=\"$VAL\""; done < /proc/$GAME_PID/environ >> steam-env-vars.sh
chmod +x steam-env-vars.sh

kill -9 $GAME_PID
sleep 1

cat <<'EOT' > $DEBUG_REAL_HOME/.steam/steam/steamapps/common/Portal\ 2/portal2/cfg/config.cfg
unbindall
bind "a" "+moveleft"
bind "c" "+voicerecord"
bind "d" "+moveright"
bind "e" "+use"
bind "f" "+mouse_menu"
bind "q" "+mouse_menu_taunt"
bind "r" "swap_ss_input"
bind "s" "+back"
bind "t" "say"
bind "w" "+forward"
bind "`" "toggleconsole"
bind "SPACE" "+jump"
bind "TAB" "+remote_view"
bind "ESCAPE" "cancelselect"
bind "PAUSE" "pause"
bind "CTRL" "+duck"
bind "F3" "askconnect_accept"
bind "F5" "jpeg"
bind "F6" "save quick"
bind "F7" "load quick"
bind "MOUSE1" "+attack"
bind "MOUSE2" "+attack2"
bind "MOUSE3" "+zoom"
bind "MWHEELUP" "+zoom_in"
bind "MWHEELDOWN" "+zoom_out"
bind "JOY1" "+jump"
bind "JOY2" "+duck"
bind "JOY3" "+use"
bind "JOY4" "+remote_view"
bind "JOY5" "+quick_ping"
bind "JOY6" "+zoom"
bind "JOY8" "gameui_activate"
bind "POV_UP" "+mouse_menu_taunt"
bind "POV_LEFT" "+mouse_menu"
bind "Z AXIS POS" "+attack2"
bind "Z AXIS NEG" "+attack"
cmd2 bind "JOY1" "+jump"
cmd2 bind "JOY2" "+duck"
cmd2 bind "JOY3" "+use"
cmd2 bind "JOY4" "+remote_view"
cmd2 bind "JOY5" "+quick_ping"
cmd2 bind "JOY6" "+zoom"
cmd2 bind "JOY8" "gameui_activate"
cmd2 bind "POV_UP" "+mouse_menu_taunt"
cmd2 bind "POV_LEFT" "+mouse_menu"
cmd2 bind "Z AXIS POS" "+attack2"
cmd2 bind "Z AXIS NEG" "+attack"
adsp_debug "0"
ai_report_task_timings_on_limit "0"
ai_think_limit_label "0"
budget_averages_window "30"
budget_background_alpha "128"
budget_bargraph_background_alpha "128"
budget_bargraph_range_ms "16.6666666667"
budget_history_numsamplesvisible "100"
budget_history_range_ms "66.666666667"
budget_panel_bottom_of_history_fraction ".25"
budget_panel_height "384"
budget_panel_width "512"
budget_panel_x "0"
budget_panel_y "50"
budget_peaks_window "30"
budget_show_averages "0"
budget_show_history "1"
budget_show_peaks "1"
bugreporter_uploadasync "0"
bugreporter_username ""
c_maxdistance "200"
c_maxpitch "90"
c_maxyaw "135"
c_mindistance "30"
c_minpitch "0"
c_minyaw "-135"
c_orthoheight "100"
c_orthowidth "100"
c_thirdpersonshoulder "false"
c_thirdpersonshoulderaimdist "120.0"
c_thirdpersonshoulderdist "40.0"
c_thirdpersonshoulderheight "5.0"
c_thirdpersonshoulderoffset "20.0"
cam_collision "1"
cam_idealdelta "4.0"
cam_idealdist "150"
cam_idealdistright "0"
cam_idealdistup "0"
cam_ideallag "4.0"
cam_idealpitch "0"
cam_idealyaw "0"
cam_snapto "0"
cc_lang ""
cc_linger_time "1.0"
cc_predisplay_time "0.25"
cc_subtitles "0"
chet_debug_idle "0"
cl_allowdownload "1"
cl_allowupload "1"
cl_auto_taunt_pip "1"
cl_autowepswitch "1"
cl_chatfilters "31"
cl_class "default"
cl_cmdrate "30"
cl_debugrumble "0"
cl_disable_survey_panel "0"
cl_downloadfilter "all"
cl_forcepreload "0"
cl_idealpitchscale "0.8"
cl_ignore_vpk_assocation "0"
cl_logofile "materials/vgui/logos/spray_bullseye.vtf"
cl_minimal_rtt_shadows "1"
cl_mouselook "1"
cl_mouselook2 "1"
cl_npc_speedmod_intime "0.25"
cl_npc_speedmod_outtime "1.5"
cl_observercrosshair "1"
cl_photo_disable_model_alpha_writes "1"
cl_playermodel "models/player/chell/player.mdl"
cl_playerspraydisable "0"
cl_rumblescale "1.0"
cl_rumblescale2 "1.0"
cl_showbackpackrarities "0"
cl_showhelp "1"
cl_showpluginmessages "1"
cl_skip_player_render_in_main_view "1"
cl_soundfile ""
cl_support_vpk_assocation "0"
cl_team "default"
cl_thirdperson "0"
cl_timeout "30"
cl_updaterate "20"
closecaption "0"
cm_max_history_chambers "500"
cm_max_quickplay_maps "50"
cm_play_intro_video "0"
con_enable "1"
crosshair "1"
dsp_enhance_stereo "0"
dsp_slow_cpu "0"
dsp_volume "0.8"
force_audio_english "0"
fov_desired "90.000000"
func_break_max_pieces "15"
g15_update_msec "250"
gameinstructor_enable "1"
hud_draw_fixed_reticle "1"
hud_fastswitch "0"
hud_quickinfo "1"
hud_quickinfo_swap "0"
hud_takesshots "0"
joy_accelmax "1.0"
joy_accelscale "2.0"
joy_advanced "1"
joy_advaxisr "2"
joy_advaxisu "4"
joy_advaxisv "0"
joy_advaxisx "3"
joy_advaxisy "1"
joy_advaxisz "0"
joy_autoaimdampen "0.5"
joy_autoaimdampenrange "0.85"
joy_axis_deadzone "0.2"
joy_axisbutton_threshold "0.3"
joy_cfg_custom_bindingsA "0"
joy_cfg_custom_bindingsA2 "0"
joy_cfg_custom_bindingsB "0"
joy_cfg_custom_bindingsB2 "0"
joy_cfg_preset "1"
joy_cfg_preset2 "1"
joy_circle_correct "1"
joy_diagonalpov "0"
joy_display_input "0"
joy_forwardsensitivity "-1"
joy_forwardthreshold "0.15"
joy_gamecontroller_config ""
joy_invertx "0"
joy_invertx2 "0"
joy_inverty "0"
joy_inverty2 "0"
joy_legacy "0"
joy_legacy2 "0"
joy_lowend "0.75"
joy_lowmap "0.25"
joy_movement_stick "0"
joy_movement_stick2 "0"
joy_name "Xbox360 controller"
joy_no_accel_jump "0"
joy_pitchsensitivity "0.75"
joy_pitchsensitivity2 "1"
joy_pitchthreshold "0.15"
joy_response_look "1"
joy_response_move "5"
joy_sensitive_step0 "0.1"
joy_sensitive_step1 "0.4"
joy_sensitive_step2 "0.90"
joy_sidesensitivity "1"
joy_sidethreshold "0.15"
joy_vibration "1"
joy_vibration2 "1"
joy_wingmanwarrior_turnhack "0"
joy_yawsensitivity "-1.5"
joy_yawsensitivity2 "-1"
joy_yawthreshold "0.15"
joystick "1"
lookspring "0"
lookstrafe "0"
m_customaccel "0"
m_customaccel_exponent "1.3"
m_customaccel_max "0"
m_customaccel_scale "0.04"
m_forward "1"
m_mouseaccel1 "0"
m_mouseaccel2 "0"
m_mousespeed "1"
m_pitch "0.022"
m_pitch2 "0.022"
m_rawinput "0"
m_side "0.8"
m_yaw "0.022"
mat_monitorgamma "2.200000"
mat_monitorgamma_tv_enabled "0"
mat_powersavingsmode "0"
mat_spewalloc "0"
mat_vsync "0"
mm_server_search_lan_ports "27015,27016,27017,27018,27019,27020"
move_during_ui "false"
mp_auto_accept_team_taunt "1"
muzzleflash_light "1"
name "phoronix"
name2 "unnamed"
net_allow_multicast "1"
net_graph "0"
net_graphheight "64"
net_graphmsecs "400"
net_graphpos "1"
net_graphproportionalfont "1"
net_graphshowinterp "1"
net_graphshowlatency "1"
net_graphsolid "1"
net_graphtext "1"
net_maxroutable "1200"
net_scale "5"
net_steamcnx_allowrelay "1"
npc_height_adjust "1"
overview_alpha "1.0"
overview_health "1"
overview_locked "1"
overview_names "1"
overview_tracks "1"
password ""
portal_demohack "0"
puzzlemaker_current_hint "0"
puzzlemaker_drawselectionmeshes "0"
puzzlemaker_enable_budget_bar "0"
puzzlemaker_play_sounds "1"
puzzlemaker_shadows "0"
puzzlemaker_show_budget_numbers "0"
puzzlemaker_zoom_to_mouse "1"
r_drawmodelstatsoverlaymax "1.5"
r_drawmodelstatsoverlaymin "0.1"
r_eyegloss "1"
r_eyemove "1"
r_eyeshift_x "0"
r_eyeshift_y "0"
r_eyeshift_z "0"
r_eyesize "0"
r_paintblob_calc_color "0"
r_paintblob_calc_hifreq_color "0"
r_paintblob_calc_tan_only "0"
r_paintblob_calc_tile_color "0"
r_paintblob_calc_uv_and_tan "1"
r_portal_stencil_depth "2"
sc_debug_sets "0"
sc_enable "1.0"
sc_enable2 "1.0"
sc_joystick_inner_deadzone_scale "1"
sc_joystick_map "1"
sc_joystick_outer_deadzone_scale "0.75"
sc_pitch_sensitivity_new2 "0.10"
sc_pitch_sensitivity_new22 "0.10"
sc_yaw_sensitivity_new2 "0.10"
sc_yaw_sensitivity_new22 "0.10"
scene_showfaceto "0"
scene_showlook "0"
scene_showmoveto "0"
scene_showunlock "0"
sdl_displayindex "0"
sensitivity "3"
sk_autoaim_mode "2"
skill "1.000000"
snd_duckerattacktime "0.5"
snd_duckerreleasetime "2.5"
snd_duckerthreshold "0.15"
snd_ducking_off "1"
snd_ducktovolume "0.55"
snd_legacy_surround "0"
snd_mixahead "0.1"
snd_musicvolume "1.0"
snd_mute_losefocus "1"
snd_pitchquality "1"
spec_scoreboard "0"
ss_splitmode "0"
store_version "1"
suitvolume "0.25"
sv_forcepreload "0"
sv_holdrotationsensitivity "0.1"
sv_log_onefile "0"
sv_logbans "0"
sv_logecho "1"
sv_logfile "1"
sv_logflush "0"
sv_logsdir "logs"
sv_noclipaccelerate "5"
sv_noclipspeed "5"
sv_pvsskipanimation "1"
sv_skyname "sky_white"
sv_specaccelerate "5"
sv_specnoclip "1"
sv_specspeed "3"
sv_unlockedchapters "1"
sv_voiceenable "1"
texture_budget_background_alpha "128"
texture_budget_panel_bottom_of_history_fraction ".25"
texture_budget_panel_height "284"
texture_budget_panel_width "512"
texture_budget_panel_x "0"
texture_budget_panel_y "450"
tf_explanations_backpackpanel "1"
tv_nochat "0"
ui_lastact_played "1"
ui_public_lobby_filter_campaign ""
ui_public_lobby_filter_difficulty2 ""
ui_public_lobby_filter_status ""
viewmodel_offset_x "0.0"
viewmodel_offset_y "0.0"
viewmodel_offset_z "0.0"
voice_enable "1"
voice_forcemicrecord "1"
voice_modenable "1"
voice_scale "1"
voice_threshold "2000"
voice_vox "1"
volume "1.0"
vprof_graphheight "256"
vprof_graphwidth "512"
vprof_unaccounted_limit "0.3"
vprof_verbose "1"
vprof_warningmsec "10"
xbox_autothrottle "1"
xbox_throttlebias "100"
xbox_throttlespoof "200"
cmd1 +jlook
cmd2 +jlook
EOT
