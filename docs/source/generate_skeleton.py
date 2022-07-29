import os

import deimos
import sphinx_skeleton as ss

if __name__ == "__main__":
    # generic package reference
    package = deimos

    # modules
    modules = ss.get_modules(os.path.dirname(package.__file__))

    # documentation layout
    doc_layout = {'getting_started': ['installation',
                                      'tutorial',
                                      'cli'],
                  # edit user_guide per package
                  'user_guide': ['loading_saving',
                                 'peak_detection',
                                 'ccs_calibration',
                                 'ms2_extraction',
                                 'isotope_detection',
                                 'extracted_ion'],
                  'api_reference': [package.__name__] + modules,
                  'project_info': ['faq',
                                   'citing_and_citations',
                                   'contributing',
                                   'license',
                                   'disclaimer']}

    # readable titles
    titles = {'api_reference': 'API Reference',
              'getting_started': 'Getting Started',
              'project_info': 'Project Info',
              'user_guide': 'User Guide',
              'installation': 'Installation',
              'tutorial': 'Tutorial',
              'cli': 'Command Line Interface',
              'loading_saving': 'Loading/Saving',
              'peak_detection': 'Peak Detection',
              'ccs_calibration': 'CCS Calibration',
              'ms2_extraction': 'MS2 Extraction',
              'isotope_detection': 'Isotope Detection',
              'extracted_ion': 'Extracted Ion Functionality',
              'citing_and_citations': 'Citing and Citations',
              'faq': 'Frequently Asked Questions',
              'contributing': 'Contributing',
              'license': 'License',
              'disclaimer': 'Disclaimer'}

    # license
    with open(os.path.join(os.path.dirname(os.path.dirname(package.__file__)), 'LICENSE')) as f:
        license = f.read()

    # instance
    skel = ss.SphinxSkeletonizer(package)
    skel.set_tree(doc_layout)
    skel.set_titles(titles)
    skel.clean()

    # custom write functions
    def api_reference_fxn(key):
        if package.__name__ not in key:
            fullkey = '{}.{}'.format(package.__name__, key)
        else:
            fullkey = key

        string = ss.generate_title(key)
        string += '.. automodule:: {}\n'.format(fullkey)
        string += '\t:members:\n'
        string += '\t:private-members:\n'
        string += '\t:undoc-members:\n'

        # top level
        if package.__name__ in key:
            string += '\t:imported-members:\n'

        return string

    def license_fxn(key):
        string = ss.generate_title(key)
        string += license
        return string

    def citation_fxn(key):
        string = ss.generate_title(key)
        string += ('If you would like to reference {} in an academic paper,'
                   'we ask you include the following:\n'
                   '[citations here]\n').format(package.__name__)
        return string

    # set write functions
    for page in doc_layout['api_reference']:
        skel.set_template(page, api_reference_fxn)

    skel.set_template('license', license_fxn)
    skel.set_template('citing_and_citations', citation_fxn)

    skel.build_tree()
    skel.write_index()
