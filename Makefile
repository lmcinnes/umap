# make gh-pages in repo base directory to automatically build and deploy documents to github

gh-pages:
	echo "Make gh-pages"
	cd doc; make html
	git checkout gh-pages
	rm -rf _sources _static _modules _downloads _images auto_examples
	mv -fv doc/_build/html/* .
	rm -rf doc
	git add -A
	git commit -m "Generated gh-pages for `git log master -1 --pretty=short --abbrev-commit`" && git push origin gh-pages ; git checkout master
