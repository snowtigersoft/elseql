#!/usr/bin/env python

from __future__ import print_function
import json
import datetime
from dateutil import parser
from pyparsing import *
from delorean import parse
import time


class Operator(object):
    name = '<UnknownOperator>'

    def __repr__(self):
        return "(%s %s)" % (self.name, self.operands)

    def __init__(self, operands):
        self.operands = operands

    def op(self, i):
        return self.val(self.operands[i])

    def val(self, x):
        if isinstance(x, basestring):
            return x  # escape Lucene characters?
        elif isinstance(x, bool):
            return "true" if x else "false"
        else:
            return str(x)

class BinaryOperator(Operator):
    def __init__(self, operands):
        self.name = operands[1]
        self.operands = [ operands[0], operands[2] ]

    def __str__(self):
        if self.name == '=':
            return "%s:%s" % (self.operands[0], self.op(1))
        elif self.name == '!=' or self.name == '<>':
            return "NOT (%s:%s)" % (self.operands[0], self.op(1))
        elif self.name in ['<=', 'LTE', 'LE']:
            return "%s:[* TO %s]" % (self.operands[0], self.op(1))
        elif self.name in ['>=', 'GTE', 'GE']:
            return "%s:[%s TO *]" % (self.operands[0], self.op(1))
        elif self.name in ['<', 'LT']:
            return "%s:[* TO %s]" % (self.operands[0], self.op(1))
        elif self.name in ['>', 'GT']:
            return "%s:[%s TO *]" % (self.operands[0], self.op(1))
        else:
            return "%s %s %s" % (self.operands[0], self.name, self.op(1))

class LikeOperator(Operator):
    name = 'LIKE'

    def __str__(self):
        return "%s:%s" % (self.operands[0], self.operands[1].replace('*','\*').replace('%','*'))

class BetweenOperator(Operator):
    name = 'BETWEEN'

    def __str__(self):
        return "%s:[%s TO %s]" % (self.operands[0], self.op(1), self.op(2))

class InOperator(Operator):
    name = 'IN'

    def __init__(self, operands):
        self.operands = [operands[0], operands[1:]]

    def __str__(self):
        return "%s:(%s)" % (self.operands[0], ' OR '.join([self.val(x) for x in self.operands[1]]))

class AndOperator(Operator):
    def __init__(self, operands=None):
        self.name = 'AND'
        self.operands = [x for x in operands[0] if not isinstance(x, basestring)]

    def __str__(self):
        return '(%s)' % ' AND '.join([self.val(x) for x in self.operands])

class OrOperator(Operator):
    def __init__(self, operands=None):
        self.name = 'OR'
        self.operands = [x for x in operands[0] if not isinstance(x, basestring)]

    def __str__(self):
        return '(%s)' % ' OR '.join([self.val(x) for x in self.operands])

class NotOperator(Operator):
    def __init__(self, operands=None):
        self.name = 'NOT'
        self.operands = [operands[0][1]]

    def __str__(self):
        return "NOT %s" % self.operands[0]

class QueryFilter(Operator):
    def __init__(self, operands=None):
        self.name = "query"
        self.operands = [ operands[0] ]

    def __str__(self):
        return self.operands[0]

class ExistFilter(Operator):
    def __init__(self, operands=None):
        self.name = "exists"
        self.operands = [ operands[0] ]

    def __str__(self):
        return self.operands[0]

class MissingFilter(Operator):
    def __init__(self, operands=None):
        self.name = "missing"
        self.operands = [ operands[0] ]

    def __str__(self):
        return self.operands[0]

def makeGroupObject(cls):
    def groupAction(s,loc,tokens):
        return cls(tokens)
    return groupAction

def invalidSyntax(s, loc, token):
    raise ParseFatalException(s, loc, "Invalid Syntax")

def intValue(t):
    return int(t)

def floatValue(t):
    return float(t)

def timestampValue(t):
    return int(time.mktime(parser.parse(t).timetuple()))

def boolValue(t):
    return t.lower() == 'true'

def function_now(t):
    dt = datetime.datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def function_cur_date(t):
    dt = datetime.datetime.now()
    return dt.strftime("%Y-%m-%d")

def function_data_add(t):
    if not isinstance(t, ParseResults):
        raise TypeError('should be ParseResults type')
    cur_date, num, tp = t[0], t[1], t[2]
    d = parse(cur_date).__getattr__('next_%s' % tp.lower())(int(num))
    return d.datetime.strftime("%Y-%m-%d %H:%M:%S")

def function_data_sub(t):
    if not isinstance(t, ParseResults):
        raise TypeError('should be ParseResults type')
    cur_date, num, tp = t[0], t[1], t[2]
    d = parse(cur_date).__getattr__('last_%s' % tp.lower())(int(num))
    return d.datetime.strftime("%Y-%m-%d %H:%M:%S")

def makeAtomObject(fn):
    def atomAction(s, loc, tokens):
        try:
            return fn(tokens[0])
        except TypeError:
            try:
                return fn(tokens)
            except Exception as e:
                raise ParseFatalException(s, loc, e.message)
        except Exception as e:
            raise ParseFatalException(s, loc, e.message)
    return atomAction


def facet_type(type_name):
        return CaselessKeyword(type_name.lower()).setResultsName('type')


class ElseParserException(ParseBaseException):
    pass


class ElseParser(object):
    # define SQL tokens
    selectStmt   = Forward()
    selectToken  = CaselessKeyword("SELECT")
    facetToken   = CaselessKeyword("BROWSE BY")
    scriptToken  = CaselessKeyword("SCRIPT")
    fromToken    = CaselessKeyword("FROM")
    whereToken   = CaselessKeyword("WHERE")
    useAnalyzerToken = CaselessKeyword("USE ANALYZER")
    orderbyToken = CaselessKeyword("ORDER BY")
    limitToken   = CaselessKeyword("LIMIT")
    between      = CaselessKeyword("BETWEEN")
    likeop       = CaselessKeyword("LIKE")
    in_          = CaselessKeyword("IN")
    and_         = CaselessKeyword("AND")
    or_          = CaselessKeyword("OR")
    not_         = CaselessKeyword("NOT")

    filterToken  = CaselessKeyword("FILTER")
    queryToken   = CaselessKeyword("QUERY")
    existToken   = CaselessKeyword("EXIST")
    missingToken = CaselessKeyword("MISSING")

    nowFuncToken = CaselessKeyword("NOW")
    curDateFuncToken = CaselessKeyword("CURDATE")
    dateAddFuncToken = Suppress(CaselessKeyword("DATE_ADD"))
    dateSubFuncToken = Suppress(CaselessKeyword("DATE_Sub"))
    timestampFuncToken = Suppress(CaselessKeyword("TIMESTAMP"))

    E      = CaselessLiteral("E")
    binop  = oneOf("= >= <= < > <> != LT LTE LE GT GTE GE", caseless=True)
    lpar   = Suppress("(")
    rpar   = Suppress(")")
    comma  = Suppress(",")

    arithSign = Word("+-", exact=1)

    ident          = Word( alphas + "_", alphanums + "_$" ).setName("identifier")
    columnName     = delimitedList( ident, ".", combine=True )
    columnNameList = Group( delimitedList( columnName ) )
    indexName = Combine(delimitedList(ident + Optional("*"), ",", combine=True) + Optional("." + ident))

    #likeExpression fore SQL LIKE expressions
    likeExpr       = quotedString.setParseAction( removeQuotes )

    realNum = Combine(
        Optional(arithSign) +
        ( Word( nums ) + "." + Optional( Word(nums) ) | ( "." + Word(nums) ) ) +
        Optional( E + Optional(arithSign) + Word(nums) ) ) \
            .setParseAction(makeAtomObject(floatValue))

    intNum = Combine( Optional(arithSign) + Word( nums ) +
        Optional( E + Optional("+") + Word(nums) ) ) \
            .setParseAction(makeAtomObject(intValue))

    boolean = oneOf("true false", caseless=True) \
        .setParseAction(makeAtomObject(boolValue))

    #functipn support
    date_type = oneOf("YEAR MONTH DAY HOUR MINUTE SECOND WEEK", caseless=True)
    func_cur_date = (curDateFuncToken+lpar+rpar).setParseAction(makeAtomObject(function_cur_date))
    func_now = (nowFuncToken+lpar+rpar).setParseAction(makeAtomObject(function_now))
    func_date_date_time = func_now | func_cur_date
    func_date_add = (dateAddFuncToken+lpar+func_date_date_time+comma+intNum+comma+date_type+rpar) \
        .setParseAction(makeAtomObject(function_data_add))
    func_date_sub = (dateSubFuncToken+lpar+func_date_date_time+comma+intNum+comma+date_type+rpar) \
        .setParseAction(makeAtomObject(function_data_sub))
    date_funcs = func_date_date_time | func_date_add | func_date_sub

    unix_timestamp = (timestampFuncToken + lpar + (quotedString | date_funcs) + rpar) \
        .setParseAction(makeAtomObject(timestampValue))

    columnRval = realNum | intNum | boolean | unix_timestamp | date_funcs | quotedString.setParseAction(removeQuotes)

    whereCondition = ( columnName + binop + columnRval ) \
            .setParseAction(makeGroupObject(BinaryOperator)).setResultsName('term') \
       | ( columnName + in_.suppress() + lpar + delimitedList( columnRval ) + rpar ).setParseAction(makeGroupObject(InOperator)) \
       | ( columnName + between.suppress() + columnRval + and_.suppress() + columnRval ).setParseAction(makeGroupObject(BetweenOperator)) \
       | ( columnName + likeop.suppress() + likeExpr  ).setParseAction(makeGroupObject(LikeOperator))

    boolOperand = whereCondition | boolean

    whereExpression = quotedString.setParseAction( removeQuotes ) \
        | operatorPrecedence( boolOperand,
            [
                (not_, 1, opAssoc.RIGHT, NotOperator),
                (or_,  2, opAssoc.LEFT,  OrOperator),
                (and_, 2, opAssoc.LEFT,  AndOperator),
            ])

    filterExpression = (queryToken.suppress() + whereExpression.setResultsName("query")).setParseAction(makeGroupObject(QueryFilter)) \
        | (existToken.suppress() + columnName).setParseAction(makeGroupObject(ExistFilter)) \
        | (missingToken.suppress() + columnName).setParseAction(makeGroupObject(MissingFilter))

    #facets
    facetGlobal = Optional((comma + boolean.setResultsName('is_global')), default=False)
    facetSize = Optional(comma + intNum.setResultsName('size'), default=10)
    columnNameOrScript = columnName.setResultsName('field') | quotedString.setResultsName('script_field')
    columnNameOrScriptOrColumns = columnNameOrScript \
        | (lpar + Group(delimitedList(columnName)).setResultsName('fields') + rpar)

    #terms facet
    facetTermsOrder = oneOf("count term reverse_count reverse_term", caseless=True)
    facetTermsType = facet_type("TERMS") + comma + columnNameOrScriptOrColumns + facetSize \
        + Optional(comma + facetTermsOrder.setResultsName('order')) + facetGlobal
    #terms states facet
    facetTermStatesOrder = oneOf("term reverse_term count reverse_count total reverse_total min reverse_min max reverse_max mean reverse_mean", caseless=True)
    facetTermStatesType = facet_type("TERMS_STATS") + comma + columnName.setResultsName('key_field') \
        + comma + columnNameOrScript.setResultsName('value_field') + facetSize \
        + Optional(comma + facetTermsOrder.setResultsName('order')) + facetGlobal
    #statistical facet
    facetStatisticalType = facet_type("STATISTICAL") + comma \
        + columnNameOrScriptOrColumns + facetGlobal

    facetTypes = facetTermsType | facetTermStatesType | facetStatisticalType
    facetExpression = quotedString.setParseAction(removeQuotes) \
        | Group(delimitedList(Group(columnName.setResultsName('facet_name') + Optional(lpar + facetTypes + rpar))))

    orderseq  = oneOf("asc desc", caseless=True)
    orderList = delimitedList(Group(columnName + Optional(orderseq, default="asc")))

    limitoffset = intNum
    limitcount  = intNum

    #selectExpr  = ( 'count(*)' | columnNameList | '*' )
    selectExpr = (columnNameList | '*' | (CaselessKeyword('COUNT') + lpar + '*' + rpar))
    scriptExpr = columnName + Suppress("=") + quotedString.setParseAction( removeQuotes )

    # define the grammar
    selectStmt << (selectToken +
        selectExpr.setResultsName("fields") +
        Optional(scriptToken + scriptExpr.setResultsName("script")) +
        fromToken + indexName.setResultsName("index") +
        Optional(whereToken + whereExpression.setResultsName("query")) +
        Optional(useAnalyzerToken + ident.setResultsName("analyzer")) +
        Optional(filterToken + filterExpression.setResultsName("filter")) +
        Optional(orderbyToken + orderList.setResultsName("order")) +
        Optional(limitToken + Group(Optional(limitoffset + comma) + limitcount).setResultsName("limit")) +
        Optional(facetToken + facetExpression.setResultsName("facets"))
    )

    grammar_parser = selectStmt

    @staticmethod
    def parse(stmt, debug=False):
        ElseParser.grammar_parser.setDebug(debug)

        try:
            return ElseParser.grammar_parser.parseString(stmt, parseAll=True)
        except (ParseException, ParseFatalException) as err:
            raise ElseParserException(err.pstr, err.loc, err.msg, err.parserElement)

    @staticmethod
    def test(stmt):
        print("STATEMENT: ", stmt)
        print()

        try:
            response = ElseParser.parse(stmt)
            print(str(response.query).find('created_at'))
            print("index  = ", response.index)
            print("fields = ", response.fields)
            print("query  = ", response.query)
            print("analyzer  = ", response.analyzer)
            print("script = ", response.script)
            print("filter = ", response.filter)
            print("order  = ", response.order)
            print("limit  = ", response.limit)
            print("facets = ", response.facets)
            print("FACETS Parsed:")
            if response.query:
                data = {'query': {'query_string': {'query': str(response.query), 'default_operator': 'AND'}}}
                if response.analyzer:
                    data['query']['query_string']['analyzer'] = response.analyzer
            else:
                data = {'query': {'match_all': {}}}
            data['facets'] = {}
            for f in response.facets:
                if isinstance(f, basestring):
                    data['facets'] = json.loads(str(f))
                elif f.type == '':
                    data['facets'][f.facet_name] = {
                        'terms': {
                            "field": f.facet_name
                        }
                    }
                else:
                    tmp = {f.type: {}}
                    if f.type == 'terms':
                        if f.field:
                            tmp[f.type]['field'] = f.field
                        elif f.fields:
                            tmp[f.type]['fields'] = list(f.fields)
                        if f.script_field:
                            tmp[f.type]['script_field'] = f.script_field
                        if f.size:
                            tmp[f.type]['size'] = f.size
                        if f.order:
                            tmp[f.type]['order'] = f.order
                    elif f.type == 'terms_stats':
                        tmp[f.type]['key_field'] = f.key_field
                        if f.script_field:
                            tmp[f.type]['value_script'] = f.script_field
                        else:
                            tmp[f.type]['value_field'] = f.value_field
                        if f.size:
                            tmp[f.type]['size'] = f.size
                        if f.order:
                            tmp[f.type]['order'] = f.order
                    elif f.type == 'statistical':
                        if f.field:
                            tmp[f.type]['field'] = f.field
                        elif f.fields:
                            tmp[f.type]['fields'] = list(f.fields)
                        if f.script_field:
                            tmp[f.type]['script'] = f.script_field
                    if f.is_global:
                        tmp[f.type]['global'] = f.is_global

                    data['facets'][f.facet_name] = tmp
            print(json.dumps(data))

        except ElseParserException as err:
            print(err.pstr)
            print(" "*err.loc + "^\n" + err.msg)
            print("ERROR:", err)

        print()

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        stmt = " ".join(sys.argv[1:])
    else:
        #stmt = "select * from user_*,test,weibo_statu_*.type where (text like '%cool and warm' or  gender in ('m')) and city=1 and created_at>TIMESTAMP('2013') or created_at between TIMESTAMP('2013-10-14') and TIMESTAMP('2014') and user_prov between '*' and 90 use analyzer ik order by uid limit 2,10  browse by cid1(terms,cid, 10, count, true), cid2(terms_stats, cid, cid, 10, count, false), st(STATISTICAL, (cid, user_prov) , true)"
        #test function
        stmt = "select * from user where created_time<timestamp(curdate()) and cc=timestamp(date_sub(curdate(), 1, DAY)) and dd=now()"
    ElseParser.test(stmt)
