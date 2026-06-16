from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from urllib.parse import urlparse

import html
import yaml


def is_safe_url(url):
    allowed_schemes = {"http", "https", "mailto"}

    parsed = urlparse(url)

    return (
        parsed.scheme in allowed_schemes
        or url == ""
    )


def render_link(label, url=""):
    label = html.escape(str(label))
    url = str(url or "").strip()

    if not is_safe_url(url):
        url = ""

    url = html.escape(url)

    if url:
        return f'<a href="{url}">{label}</a>'

    return label


def render_name(row):
    name_field = row.get("name", "")
    url = row.get("url", "")

    if isinstance(name_field, list):
        parts = []
        for item in name_field:
            label = item.get("label", item.get("name", ""))
            item_url = item.get("url", "")
            parts.append(render_link(label, item_url))
        return " / ".join(parts)

    return render_link(name_field, url)


def sort_key(row):
    name_field = row.get("name", "")

    if isinstance(name_field, list) and name_field:
        first = name_field[0].get("label", name_field[0].get("name", ""))
        return str(first).lower()

    return str(name_field or "").lower()


def render_qmc_software_table(data_path, mode="web", start=0, stop=None, return_string=False):
    if return_string:
        buffer = StringIO()
        with redirect_stdout(buffer):
            render_qmc_software_table(
                data_path=data_path,
                mode=mode,
                start=start,
                stop=stop,
                return_string=False,
            )
        return buffer.getvalue()

    data_path = Path(data_path)

    with data_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(
            f"Expected YAML file at {data_path} "
            f"to contain a top-level list of mappings."
        )
    
    data = sorted(data, key=sort_key)
    data = data[start:stop]

    show_description = mode == "web"
    show_contact = mode == "web"

    print("""
<div class="table-responsive">
<table class="qmc-software-table">
<thead>
<tr>
    <th style="width: 48%;">Name</th>
    <th style="width: 14%;">Language</th>
    <th style="width: 20%;">Development Status</th>
""")

    if show_contact:
        print('  <th style="width: 18%;">Contact</th>')

    print("""
</tr>
</thead>
<tbody>
""")

    for row in data:
        name_html = render_name(row)

        language = html.escape(row.get("language", ""))
        raw_status = html.escape(row.get("status", ""))
        desc = html.escape(" ".join(str(row.get("description", "")).split()))

        if mode == "web":
            status = raw_status.replace(
                ", Collaboration welcome",
                "<br>Collaboration welcome",
            )
        else:
            status = raw_status

        related = row.get("related", [])

        if related:
            related_links = []

            for r in related:
                rname = r.get("name", "")
                rurl = r.get("url", "")
                related_links.append(render_link(rname, rurl))

            related_html = ", ".join(related_links)

            if mode == "web":
                name_html += (
                    f'<br><span class="software-related">'
                    f'Related: {related_html}'
                    f'</span>'
                )
            else:
                name_html += (
                    f' <span class="software-related">'
                    f'(also {related_html})'
                    f'</span>'
                )

        if show_description and desc:
            name_html += f'<br><span class="software-desc">{desc}</span>'

        row_html = f"""
<tr>
  <td>{name_html}</td>
  <td>{language}</td>
  <td><span class="status-nowrap">{status}</span></td>
"""

        if show_contact:
            contacts = row.get("contact", [])
            contact_items = []

            for c in contacts:
                if isinstance(c, dict):
                    cname = c.get("name", "")
                    curl = c.get("url", "")

                    if curl and str(curl).startswith("mailto:"):
                        label = f"✉ {cname}"
                    else:
                        label = cname

                    contact_items.append(render_link(label, curl))

                else:
                    contact_items.append(html.escape(str(c)))

            contact_str = "<br>".join(contact_items)
            row_html += f"  <td>{contact_str}</td>\n"

        row_html += "</tr>"

        print(row_html)

    print("""
</tbody>
</table>
</div>
""")
