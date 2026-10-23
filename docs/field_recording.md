# One field recording to replace the assumed noise numbers

The silent body numbers of [Harbor](harbor.md) depend on the noise model ([Noise](noise.md)), and several numbers of
the `harbor_v2` model are assumed: the snapping shrimp's rate per metre of quay and the share of a click's energy in
the band the system uses, the quay machinery's level, the fibre's drift, and the interrogator's floor and its shape.
This page says what single recording, made in a working harbor, would replace them with measured numbers. It is written
to be handed to whoever can make it.

**In one paragraph.** Record two hours of the harbor's DAS cable and one hydrophone strapped to it, at a sample rate of
at least 6 kHz, in three periods: a quiet night hour, an hour of quay work, and the passage of a known vessel with its
AIS log; give the cable's route, the interrogator's settings and the hydrophone's position on the cable, the water's
temperature and depth at the site, and the times in UTC. From the night hour come the shrimp's click rate and band
energy, which moved the silent bodies most of all the assumed numbers, and the interrogator's floor and fading; from
the work hour the machinery's lines and level; from the drift below 1 Hz over the whole record the fibre's drift; from
the vessel the coupling of the water's pressure into the fibre, which the model does not have. What it cannot settle is listed at the end.

## The equipment

* **Interrogator and fibre.** A phase sensitive OTDR interrogator (make, model, firmware) on the harbor's existing
  telecom fibre or a cable laid on the seabed along the quay; single mode fibre, its type and its cable construction (loose
  tube or tight buffered, armour), and how it lies (buried, on the seabed, clamped to the quay wall). Record the gauge
  length (2 m is what the recordings use; 1 to 10 m is acceptable, but state it), the channel spacing (2 m as
  simulated, or the interrogator's native spacing), the pulse rate, the laser's linewidth if known, and whether the
  output is strain or strain rate.
* **The route.** The cable's waypoints in a local east / north frame [m] (from the as laid survey or a GNSS walk of
  the quay), with the channel number at each waypoint, so every channel has a position to within a channel spacing.
* **A co located hydrophone.** One calibrated hydrophone (its sensitivity in V/Pa and its frequency response), fixed
  to the cable within a metre of a stated channel, at least 50 m from either end of the cable and away from the quay
  machinery, recorded on the same clock. It ties the cable's units to pascals: without it the fibre's numbers stay in
  trace units.
* **Clock.** Both systems on GPS time, the offset between them stated; an AIS receiver logging every message (NMEA
  with the receiver's timestamps) for the whole recording.

## The recording

* **Sample rate and band.** At least 6 kHz for every channel (the 0.5 m recordings use 6 kHz and a processing band
  of 100 to 760 Hz); 20 kHz or more on the hydrophone so that a shrimp click's spectrum above the band is seen as well.
  No filter inside 1 Hz to 2 kHz on the cable; state any anti alias filter.
* **Three periods, about two hours in all:**
  1. a quiet night hour (between 01:00 and 04:00 local time, no vessel moving within 1 km by AIS, quay work stopped);
  2. a working hour with the quay machinery running (cranes, pumps, generators), with a note of which machines run
     when and where they stand on the quay;
  3. a period with a known vessel movement: one vessel arriving or leaving, its AIS log kept, ideally a second
     passage of the same vessel at another range.
* **Conditions.** The water temperature and the depth at the hydrophone (a temperature logger beside it for the whole
  recording; the shrimp's rate follows the temperature), the wind speed and the sea state (a weather station's log),
  the tide.

## The metadata the system needs

The recording is converted into an `Observation` as [Architecture, Real data entry point](architecture.md#real-data-entry-point)
describes: the traces in geometry order (every cable channel in order from the first waypoint, then the hydrophone,
dead channels as zeros); `fs`; `t0` in the AIS time base; the `SensorLayout` (the waypoints, the channel spacing, the
DAS mode and the gauge length, the hydrophone's position); a `ChartEnvironment` of the harbor (water and land on a
regular grid in the same east / north frame, the depth, the nominal sound speed from the temperature and salinity);
the AIS reports in the same frame and time base; and in `meta` the instrument, the units of every channel and the
calibration of the hydrophone. No reader for the interrogator's file format exists yet; a converter to these arrays is
the first piece of work.

## The analysis, in the order it answers the parameters

The order is that of the noise sweep ([Noise, Which assumed number matters](noise.md#which-assumed-number-matters)):
the parameters that moved a silent body most come first.

1. **The shrimp** (the night hour, on the hydrophone): their rate and their energy in the band moved the silent bodies
   most. Clicks detected by their peak to median ratio; their rate, divided by the length of quay within the range the
   hydrophone hears them (from the arrival times of the same click on the cable's channels, which locates it), against
   `rate_per_m`; the energy of a click below 2 kHz over its whole band, against `band_fraction_db`; the rate against
   the water temperature over the recording.
2. **The interrogator's floor and its shape** (the night hour): it moved the multistatic second UUV from tracked to
   lost at its upper end. The noise power spectral density of every channel with nothing moving: its level and its
   rise toward low frequency against `interrogator.floor_rms`, `corner_hz` and `slope`.
3. **The machinery** (the working hour against the night hour): it moved the simple harbor's wreck and UUV. The lines
   and their harmonics, located on the quay by the matched field map, and their level at 100 m on the hydrophone,
   against `machinery.level_pa` and `line_hz`.
4. **The fibre's drift and the fading** (the whole recording): they moved nothing in the sweep, so they come last. The
   power below 1 Hz per channel, its slope and its corner, against `fibre.drift_rms`, `drift_corner_hz` and
   `drift_slope`; the fraction of channels at any time whose floor stands 10 dB above the median and how long each
   stays there, against `fade_fraction` and `fade_s`.
5. **The coupling and the propagation** (the vessel). The vessel located on the map as the system does it; its level
   on the hydrophone and on the channel beside it, which gives the ratio of the fibre's response to the water's
   pressure; the arrival times against its AIS positions, which give the propagation error of the chart's sound speed.

## What this recording cannot settle

* **Season.** One recording is one temperature: the shrimp's yearly cycle (a factor of about twenty between summer and
  winter at one reef) needs the recording repeated in another season.
* **Another harbor.** The shrimp's rate per metre and the machinery depend on the harbor; the numbers are this
  harbor's.
* **Silent bodies.** No silent body of known size is in the recording unless one is placed (a moored hull of known
  dimensions within the cable's coverage would be the cheapest), so the echo chain's numbers are not tested by it.
* **The model's own limits.** It is a 2D model with one sound speed: the recording measures the noise, but not how far
  the depth and the bottom change the field, which the model cannot represent.
* **The surface noise and the quay traffic** are quoted from published curves, not measured in this harbor; the
  recording would measure them too, but they were not assumed numbers, so they are not in the order above.
