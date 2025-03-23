---
permalink: /publications/
author_profile: true
---
## Publications

### Machine learning research

{% include base_path %}

<ol>{% for post in site.publications reversed %}{% if post.category == 'ML' %}{% include achieve_paper_title.html %} {% endif %}{% endfor %}</ol>

### Other artificial intelligence research

{% include base_path %}

<ol>{% for post in site.publications reversed %}{% if post.category != 'ML' %}{% include achieve_paper_title.html %} {% endif %}{% endfor %}</ol>

For details, please visit my [Google Scholar](https://scholar.google.com/citations?user=qQuCvmQAAAAJ) pages. 


