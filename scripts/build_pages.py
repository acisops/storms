import json
from pathlib import Path, PurePath
import re
import jinja2

p = Path(PurePath(__file__).parent / "json")
fns = list(p.glob("*.json"))
fns.sort()

all = []
yes = []
no = []

for fn in fns:

    with open(fn, "r") as f:
        inputs = json.load(f)

    skip = inputs.get("skip", False)
    if skip: 
        continue

    link = fn.name.split(".")[0].replace("_", "-")
    all.append(link)

    if inputs["shutdown"] == "YES":
        yes.append(link)
    else:
        no.append(link)

doc_dir = Path("doc/source/")

cat_template_file = 'category_template.rst'

for which, pages in zip(["all", "yes", "no"], [all, yes, no]):
    cat_template = open(Path("templates") / cat_template_file).read()
    cat_template = re.sub(r' %}\n', ' %}', cat_template)
    context = {"which": which,
               "pages": pages}

    outname = {
        "all": "all.rst", 
        "yes": "shutdown.rst",
        "no": "not_shutdown.rst"
    }[which]

    outfile = doc_dir / outname

    template = jinja2.Template(cat_template)

    with open(outfile, "w") as f:
        f.write(template.render(**context))

