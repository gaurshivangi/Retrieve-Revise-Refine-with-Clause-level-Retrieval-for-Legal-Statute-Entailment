import json
import re
import xml.etree.ElementTree as Et


def load_samples_from_file(xml_file):
    def _parse_article_fix(article_text):
        return re.findall('(?<=^Article )([^ \n]+)', article_text, re.MULTILINE)
    
    tree = Et.parse(xml_file)
    root = tree.getroot()
    samples = dict()
    for i in range(0, len(root)):
        for _, e in enumerate(root[i]):
            if e.tag == "t1":
                rel_article_ids = _parse_article_fix(e.text.strip())
            elif e.tag == "t2":
                question = e.text.strip()
                query = question if len(question) > 0 else None
        if query is not None:
            sample_id = root[i].attrib['id']
            sample = {
                # 'label': root[i].attrib.get('label', "N"), 
                    #   'rel_article_ids': rel_article_ids,
                      'query': query}
            samples[sample_id] = sample
        else:
            print("[Important warning] samples {} is ignored".format(sample))
    return samples

xml_file_path = 'data/R05_test/TestData_en.xml'
samples = load_samples_from_file(xml_file_path)

output_file_path = 'data/R05_test/task3_test.json'
with open(output_file_path, 'w') as fout:
    json.dump(samples, fout, indent=4, ensure_ascii=False)
