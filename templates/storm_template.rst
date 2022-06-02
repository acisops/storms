Storm: {{storm_date}}
---------------------

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

.. image:: protons_vs_time.png

.. image:: electrons_vs_time.png

.. image:: hrc_vs_time.png

Proton Spectra at Selected Times
++++++++++++++++++++++++++++++++

.. image:: proton_spectra.png

Scatter Plots
+++++++++++++

.. image:: scatter_plots.png




