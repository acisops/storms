{% if which == "all" %}
All Storms
==========
{% elif which == "yes" %}
Storms Which Did Cause Shutdowns
================================
{% else %}
Storms Which Did Not Cause Shutdowns
====================================
{% endif %}

{% for page in pages %}
* :ref:`{{page}}`
{% endfor %}