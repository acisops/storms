```{raw} html
<div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 0.5em;">
  <div>
    <div style="font-weight: bold; font-size: 2em; line-height: 1;">CHANDRA</div>
    <div style="font-weight: bold; font-size: 1.3em; line-height: 1.3;">X-ray Center</div>
    <div style="font-size: 0.9em; margin-top: 0.3em;">60 Garden St., Cambridge Massachusetts 02138 USA</div>
  </div>
  <div>
    <img src="../../cxc_logo.png" alt="CXC logo" style="height: 110px;">
  </div>
</div>
<hr style="border: none; border-top: 1px solid black; margin: 0 0 1em 0;">

<table style="border-collapse: collapse; margin-bottom: 1em;">
  <tr><td style="padding-right: 1em; font-weight: bold; vertical-align: top;">Date:</td><td>{{BUILD_DATE}}</td></tr>
  <tr><td style="padding-right: 1em; font-weight: bold; vertical-align: top;">From:</td><td>John ZuHone</td></tr>
  <tr><td style="padding-right: 1em; font-weight: bold; vertical-align: top;">To:</td><td>Chandra Operations Team</td></tr>
  <tr><td style="padding-right: 1em; font-weight: bold; vertical-align: top;">Subject:</td><td>Chandra Radiation Event and Shutdown in January 2026</td></tr>
  <tr><td style="padding-right: 1em; font-weight: bold; vertical-align: top;">Cc:</td><td>MSFC Project Science, CXC Director's Office</td></tr>
  <tr><td style="padding-right: 1em; font-weight: bold; vertical-align: top; font-size: 0.85em;">File:</td><td style="font-family: monospace; font-size: 0.85em;">JAN1926_memo.md</td></tr>
  <tr><td style="padding-right: 1em; font-weight: bold; vertical-align: top; font-size: 0.85em;">Version:</td><td style="font-family: monospace; font-size: 0.85em;">1.0</td></tr>
</table>
<hr style="border: none; border-top: 1px solid black; margin: 0 0 1.5em 0;">
<p style="text-align: right; margin: -0.5em 0 1em 0;">
  <a href="../../JAN1926_memo.pdf" style="display: inline-flex; align-items: center; gap: 0.4em; font-size: 1.3em; font-weight: bold; text-decoration: none;">
    <svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path d="M6,2 L14,2 L18,6 L18,22 L6,22 Z" fill="#f5f5f5" stroke="#666666" stroke-width="1"/>
      <path d="M14,2 L14,6 L18,6" fill="none" stroke="#666666" stroke-width="1"/>
      <rect x="4" y="13" width="14" height="6.5" rx="1" fill="#c62828"/>
      <text x="11" y="18" font-size="5.5" font-family="Arial, sans-serif" font-weight="bold" fill="white" text-anchor="middle">PDF</text>
    </svg>
    Download PDF version
  </a>
</p>
```

````{only} latex
```{raw} latex
\CXCletterhead
\To{Chandra Operations Team}
\From{John ZuHone}
\Subject{Chandra Radiation Event and Shutdown in January 2026}
\Cc{MSFC Project Science, CXC Director's Office}
\Date{{{BUILD_DATE}}}
\File{JAN1926\_memo.md}
\Version{1.0}
\memond
```
````

# Chandra Radiation Event and Shutdown in January 2026

## Abstract

This memo discusses the thought process that the operations team, especially the ACIS operations team, used during a
very high radiation event between January 19-22, 2026. *Chandra* was shut down via an autonomous trigger of SCS-107 via
the ACIS `txings` algorithm. In this case, the `txings` algorithm was triggered on the "FI Increasing" threshold. The
key decision points to resume science operations are reviewed, and the details of the storm are presented.

During this storm, the accumulated attenuated ACE P3 fluence was approximately $3 \times 10^7$. Thanks to the
autonomous shutdown, ACIS avoided accumulating another ~$4.22 \times 10^9$ of P3 fluence. The agreed annual budget
for this quantity is $2.0\times10^{10}$. 

## Introduction

During the days of January 19-22, 2026, *Chandra* experienced a radiation storm which resulted in the autonomous
shutdown of the spacecraft via an ACIS txings trigger of SCS-107 to protect the ACIS instrument from the damaging soft
solar wind protons that increase CTI. Though we were past solar maximum at the time, solar activity was still high.

The ACIS txings rate onboard monitoring software serves as the only available way to trigger autonomous shutdowns due to
high radiation levels. In this storm, the txings rates reached their trigger levels right before the peak of the proton
rates in nearly all channels, averting a substantial amount of radiation damage to the ACIS CCDs from soft (~100 keV)
proton fluence.

This memo discusses the properties of the storm, the radiation received in terms of the single-orbit and annual budgets,
the differences between various radiation measurements, and the response. 

All times in this memo are in UTC unless otherwise noted, and may be approximate. The various electron and proton
channel fluxes for ACE and GOES are in units of and assumed throughout when numbers for these fluxes are quoted. 

## January 18-22 2026 Detailed Timeline

* 2026:018 **Sunday January 18, 2026**

* 2026:018:00:00:00 The JAN1226 load is in progress.

* 2026:018:00:50:01 **Comm** begins (60 min).

* 2026:018:09:00:00 Earth-orbiting satellites detect an X1.9-class solar flare. A full-halo signature CME is associated with it.

* 2026:018:15:50:02 **Comm** begins (60 min).

* 2026:018:20:04:00 GOES proton channels begin a slow rise in flux, the higher energy channels rising later. An hour and
  a half later, the HRC proxy begins to rise. All increase for the next ~24 hours. 

* 2026:019 **Monday January 19, 2026**

* 2026:019:00:52:30 The JAN1926 load begins. 

* 2026:019:01:40:01 **Comm** begins (60 min).

* 2026:019:03:35:00 The HRC proxy crosses its nominal trigger threshold.

* 2026:019:05:10:00 The ACE proton channels jump sharply, and continue rising for the next 15 hours. ACE P3 is at ~1000.

* 2026:019:09:52:11 SCS-107 runs due to a txings trigger, out of comm. 

* 2026:019:10:20:00 The "strong" (S3) NOAA radiation threshold, with a 10 MeV integrated proton flux of 1000 pfu, is reached. 

* 2026:019:11:40:01 **Comm** begins (60 min). The spacecraft comes up with the science loads terminated; ACIS is safe.

* 2026:019:15:45:01 **Comm** begins (60 min).

* 2026:019:16:40:00 ACE and GOES proton channels begin to rise sharply, and will continue to do so for the next ~3 hours.

* 2026:019:19:10:00 The HRC proxy reaches a maximum value of ~5.8 $\times 10^4$.

* 2026:019:19:24:00 Space weather alerts go out indicating a CME is approaching Earth. Particle rates continue to sharply 
  increase. 

* 2026:019:20:05:00 ACE P3 hits a maximum value of ~5.6 $\times 10^5$.

* 2026:019:21:00:00 ACE and GOES proton channels begin to decline steeply from their peak values.

* 2026:020 **Tuesday January 20, 2026**

* 2026:020:03:30:01 **Comm** begins (60 min).

* 2026:020:14:15:01 **Comm** begins (90 min).

* 2026:020:23:34:38 Time of RADMON disable from the JAN1926 loads.

* 2026:021 **Wednesday January 21, 2026**

* 2026:021:06:50:56 Time of perigee.

* 2026:021:07:10:02 **Comm** begins (60 min).

* 2026:021:11:24:38 Time of RADMON enable from the JAN1926 loads.

* 2026:021:12:10:01 **Comm** begins (60 min).

* 2026:021:14:15:00 First command of the JAN2126 return-to-science loads. ACE P3 has leveled off at a value of ~1000, and will stay at this level for at least the next 24 hours. 

## Discussion

In the days leading up to January 18th, 2026, the sunspot region AR 4341 rotated into view. At 09:00, this sunspot
produced an X1.9-class solar flare, with which full-halo CME was associated. Early predictions indicated that the CME
would arrive by 06:00 UTC on January 20th. Later on in the day, the high-energy GOES proton channels began to rise, as
well as the HRC proxy which closely tracks them (see Figures {ref}`{{FIGNUM:goes_p}} <figure_goes_p>` and
{ref}`{{FIGNUM:hrc_proxy}} <figure_hrc_proxy>`). They would continue to rise for the next ~25 hours. ACIS Ops began to
monitor the radiation situation, but no telecon to consider manual actions were planned. 

In the early morning of January 19th, the HRC proxy passed its nominal trigger threshold (having no effect on operations
as the HRC shield is no longer being continuously operated). At around the same time, the ACE proton and electron
channels (see Figures {ref}`{{FIGNUM:ace_p}} <figure_ace_p>` and {ref}`{{FIGNUM:ace_e}} <figure_ace_e>`) all sharply
rose. 

All of the monitored radiation channels continued to rise through the morning of January 19th, and at 2026:019:09:52:11, 
SCS-107 ran due to a txings trigger, out of comm (see Figure {ref}`{{FIGNUM:txings}} <figure_txings>`), during Obsid 
32115. After the dump data was retrieved at the 6:40 am ET comm, it was determined that txings triggered on S2, the lone 
FI chip turned on for the observation, though given that the BI rates were climbing above their "increasing" triggering 
threshold, they may have triggered had the FI rates not done so. The storm continued, and shortly after the SCS-107 run 
the "strong" (S3) NOAA radiation threshold was reached, which SolarHam.com noted had not occurred since October of 2003. 
At the 6:40 am ET comm, the spacecraft came up with the science loads halted and ACIS safed. Since the CME had not yet 
arrived and was expected to on January 20th, the tentative plan for a return to science time was the morning of January 
21st. 

Later on in the afternoon of the 19th, the ACE and GOES channels began to rise very sharply. They continued to rise for
the next ~3 hours. Several hours later, alerts went out indicating the early arrival of the CME from the X1.9 flare. 
Shortly after this, ACE P3 hit a maximum value of ~5.6 $\times 10^5$ (see Figure {ref}`{{FIGNUM:ace_p3}} <figure_ace_p3>`), 
and all particle rates began to decline. Their decline would flatten early in the morning of January 20th, and either stay 
relatively flat or slowly declining for a few days afterward. ACE P3 would settle to a value of ~1000.

In the end, the total accumulated attenuated ACE P3 fluence for the orbit was $3 \times 10^7$. The total ACE P3 attenuated 
fluence that would have been accumulated had ACIS not been safed is ~$4.25 \times 10^9$. The txings trigger prevented the 
accumulation of this significant amount of fluence.

## Data plots for the January 2026 storm

```{raw} html
<script src="../../plotly.min.js"></script>
```

### ACE P3 Flux

In Figure {ref}`{{FIGNUM:ace_p3}} <figure_ace_p3>`, we have plotted the 5-minute averaged ACE P3 flux rate, in the usual units,
which are $\mathrm{protons~s^{-1}}$ $\mathrm{cm^{-2}~sr^{-1}~MeV^{-1}}$,
throughout the January 2026 storm. Also marked are radiation belt passages, the time of
autonomous SCS-107 execution, and comm times (the same times are marked in the rest of the radiation vs. time plots).

(figure_ace_p3)=
```{raw} html
<figure id="figure-ace-p3">
```

```{raw} html
:file: ace_p3.html
```

```{raw} html
<figcaption style="font-style: italic; font-size: 0.9em; text-align: center; margin-top: 0.5em;">Figure {{FIGNUM:ace_p3}}: ACE P3 flux during the January 2026 storm. Purple shaded regions indicate radiation belt passages; blue shaded regions mark scheduled DSN communications.</figcaption>
</figure>
```

````{only} latex
```{figure} ace_p3.png
:width: 100%

{{CAPTION:ace_p3}}
```
````

### ACE Proton Fluxes

Figure {ref}`{{FIGNUM:ace_p}} <figure_ace_p>` shows the flux from four ACE proton channels (P1, P3, P5, and P7) during the storm. Though only P3 is our proxy for damage to the ACIS CCDs, the other channels can serve as informative diagnostics.

(figure_ace_p)=
```{raw} html
<figure id="figure-ace-p">
```

```{raw} html
:file: ace_p.html
```

```{raw} html
<figcaption style="font-style: italic; font-size: 0.9em; text-align: center; margin-top: 0.5em;">Figure {{FIGNUM:ace_p}}: Flux from the ACE P1, P3, P5, and P7 proton channels during the January 2026 storm. Shaded regions and vertical lines have the same meaning as in Figure {{FIGREF:ace_p3}}.</figcaption>
</figure>
```

````{only} latex
```{figure} ace_p.png
:width: 100%

{{CAPTION:ace_p}}
```
````

### ACE Electron Fluxes

Figure {ref}`{{FIGNUM:ace_e}} <figure_ace_e>` shows the flux from two ACE electron channels (DE1 and DE4) during the
storm. These electron channels more often then not tend to rise and fall rapidly when the higher-energy GOES protons are
doing the same.

(figure_ace_e)=
```{raw} html
<figure id="figure-ace-e">
```

```{raw} html
:file: ace_e.html
```

```{raw} html
<figcaption style="font-style: italic; font-size: 0.9em; text-align: center; margin-top: 0.5em;">Figure {{FIGNUM:ace_e}}: Flux from the ACE DE1 and DE4 electron channels during the January 2026 storm. Shaded regions and vertical lines have the same meaning as in Figure {{FIGREF:ace_p3}}.</figcaption>
</figure>
```

````{only} latex
```{figure} ace_e.png
:width: 100%

{{CAPTION:ace_e}}
```
````

### GOES Proton Flux

Figure {ref}`{{FIGNUM:goes_p}} <figure_goes_p>` shows the flux from four GOES proton channels (P1, P3, P5, and P7) during the storm. These higher-energy protons are more representative of the radiation that triggers ACIS txings. GOES P5 and P7 have a steep increase in flux at the time of the txings trigger.

(figure_goes_p)=
```{raw} html
<figure id="figure-goes-p">
```

```{raw} html
:file: goes_p.html
```

```{raw} html
<figcaption style="font-style: italic; font-size: 0.9em; text-align: center; margin-top: 0.5em;">Figure {{FIGNUM:goes_p}}: Flux from the GOES P1, P3, P5, and P7 proton channels during the January 2026 storm. Shaded regions and vertical lines have the same meaning as in Figure {{FIGREF:ace_p3}}.</figcaption>
</figure>
```

````{only} latex
```{figure} goes_p.png
:width: 100%

{{CAPTION:goes_p}}
```
````

### HRC Proxy

Figure {ref}`{{FIGNUM:hrc_proxy}} <figure_hrc_proxy>` shows the HRC Shield Proxy during the storm. The HRC Anti-Coincidence Shield rates are no longer available for radiation monitoring, but had they been, they would have triggered SCS-107 at around the same time as the txings trigger.

(figure_hrc_proxy)=
```{raw} html
<figure id="figure-hrc-proxy">
```

```{raw} html
:file: hrc_proxy.html
```

```{raw} html
<figcaption style="font-style: italic; font-size: 0.9em; text-align: center; margin-top: 0.5em;">Figure {{FIGNUM:hrc_proxy}}: HRC Shield Proxy during the January 2026 storm. Shaded regions and vertical lines have the same meaning as in Figure {{FIGREF:ace_p3}}.</figcaption>
</figure>
```

````{only} latex
```{figure} hrc_proxy.png
:width: 100%

{{CAPTION:hrc_proxy}}
```
````

### txings Rates

(figure_txings)=
```{raw} html
<figure id="figure-txings">
```

```{raw} html
:file: txings.html
```

```{raw} html
<figcaption style="font-style: italic; font-size: 0.9em; text-align: center; margin-top: 0.5em;">Figure {{FIGNUM:txings}}: ACIS threshold-crossing (txings) rates during the January 2026 storm. Blue is for FI chips, orange is for BI. The horizontal lines are the increasing-rate trip thresholds for each chip type. Shaded regions and vertical lines have the same meaning as in Figure {{FIGREF:ace_p3}}.</figcaption>
</figure>
```

````{only} latex
```{figure} txings.png
:width: 100%

{{CAPTION:txings}}
```
````

## Lessons learned

## Resources

The research required for this memo, as well as the real-time response, would not have been possible without the valiant efforts of https://www.solarham.com to provide timely and accurate space weather information.

The archive of ACE data stored in ASCII tables at https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/ has gaps that are not back-filled; the full dataset can however be found in the ``ACE Browse'' archive:

ftp://mussel.srl.caltech.edu/pub/ace/browse/

The data are in HDF4 format, which can be converted to HDF5 data by use of a program `h4toh5` which I downloaded from https://www.hdfeos.org/software/h4toh5.php. A Python script, `get_ace.py`, which downloads the data and uses `h4toh5` to convert it is available on the 
HEAD LAN in `/data/acis/ace`. Instructions for downloading the data using this script and extracting the ACE proton channels are given in `/data/acis/ace/README_browse.md`.

The HRC Shield Proxy and GOES proton data are stored in HDF5 format here:

`/proj/sot/ska/data/arc/hrc_shield.h5`

Thanks to Peter Ford for providing the ACIS txings data. 
