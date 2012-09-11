elseql
======

A SQL-like command line client for elasticsearch

### USAGE

    elseql [--debug] [--port=host:port] 

### SEARCH SYNTAX

    SELECT {fields}
        [FACETS facet-fields]
        [SCRIPT script-field = 'script']
        FROM index
        [WHERE where-condition]
        [FILTER filter-condition]
        [ORDERY BY order-fields]
        [LIMIT [start,] count]

where:
    fields: '*' or comma-separated list of field names to be returned

    facet-fields: comma-separated list of fields to execute a facet query on

    script-field: name of script field, to be used in select clause
    script: ElasticSearch script

    index: index to query

    where-condition:
        {field-name} [ = != > >= < <= ] {value}
        {field-name} LIKE {value}
        {field-name} IN (value1, value2, ...)
        {field-name} BETWEEN {min-value} AND {max-value}
        NOT {where-condition}
        {where-condition} AND {where-condition}
        {where-condition} OR {where-condition}

    or where-condition:
        'query in Lucene syntax'

    filter-condition: same syntax as {where-condition} but executed as a filter

    order-fields: comma-separated list of {field-name} [ASC | DESC]

    start: start index for pagination
    count: maximum number of returned results

### INSTALLATION

With python and setuptools installed:

	sudo python setup.py install

You can also run the command without installing as:

	python elseql/elseql.py

To do this you will need the pyparsing and rawes packages installed, that are automatically installed in the previous step.

	sudo easy_install pyparsing
	sudo easy_install rawes

### SEE ALSO

    http://elasticsearch.org/
