##Rename or copy this file to config.ini

[default]
#Chromedriver Path
chromedriver=

[search]
#Search string with a maximum of 256 characters. GoogleScholar truncates the string with more than this size.
query=

##Start and End year of the Search.
##This script searches for papers year by year because Google Scholar only allows 1000 of paper by query
#ex:
#start_year=1970
#end_year=2020
##
start_year=
end_year=

##PICOC terms. The terms has to be separated by "|".
##PICOC terms are those you want to be found in the papers title, abstract, or keywords.
#ex:
#population=teste|testing this|other test
#intervention=abc|def|ghi
#...
#
##If any of the PICOC terms will not be used just leave it blank
##
[picoc]
population=
intervention=
comparison=
outcome=
context=