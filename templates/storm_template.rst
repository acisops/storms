.. _{{storm_date.replace(':', '-')}}:

{{storm_date}}
--------------

Basic Facts
===========

* Load on Spacecraft: {{load}}  
* Shutdown: {{shutdown}}  
{% if shutdown != "NO" %}
* Shutdown Trigger: {{trigger}}  
* Shutdown Time: {{shutdown_time}}  
* Startup Time: {{startup_time}}  
{% endif %}

Plots
=====

Radiation vs. Time
++++++++++++++++++

.. image:: rad_vs_time.png

Proton Spectra at Selected Times
++++++++++++++++++++++++++++++++

.. image:: proton_spectra.png

Scatter Plots
+++++++++++++

.. image:: scatter_plots.png

Proton & Electron Plots
+++++++++++++++++++++++

.. image:: ace_vs_time.png


