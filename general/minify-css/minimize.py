import cssutils
import re
import logging
import os
import time
cssutils.log.setLevel(logging.CRITICAL)

startTime = time.time()
os.system('cls')

def getFilesByExtension(ext, root):
    foundFiles = []
    for root, directories, files in os.walk(root):
        for f in files:
            if f.endswith(ext):
                # os.path.join(root, f) is the full path to the file
                foundFiles.append(os.path.join(root, f)) 
    return foundFiles


def flattenStyleSheet(sheet):
    ruleList = []
    for rule in sheet.cssRules:
        if rule.typeString == 'MEDIA_RULE':
            ruleList += rule.cssRules
        elif rule.typeString == 'STYLE_RULE':
            ruleList.append(rule)
    return ruleList


def findAllCSSClasses():
    usedClasses = {}
    # Find all used classes
    for htmlFile in htmlFiles:
        with open(htmlFile, 'r') as f:
            htmlContent = f.read()
        regex = r'class="(.*?)"'
        # re.DOTALL is needed to match newlines
        matched = re.finditer(regex, htmlContent, re.MULTILINE | re.DOTALL) 
        # matched is a list of re.Match objects
        for i in matched:
            for className in i.groups()[0].split(' '): # i.groups()[0] is the first group in the regex
                usedClasses[className] = ''
    return list(usedClasses.keys())


def translateUsedClasses(classList):
    for i, usedClass in enumerate(classList):
        for translation in translations:
            # If the class is found in the translations list, replace it
            regex = translation[0]
            subst = translation[1]
            if re.search(regex, usedClass):
                # re.sub() replaces the regex with the subst
                result = re.sub(regex, subst, usedClass, 1, re.MULTILINE) # 1 is the max number of replacements
                # Replace the class in the list
                classList[i] = result
    return classList


htmlFiles = getFilesByExtension('.html', '.')

cssFiles = getFilesByExtension('.css', 'style')

# Use Translations if the class names in the Markup dont exactly 
# match the CSS Selector ( Except for the dot at the begining. )
translations = [
    [
        '@',
        '\\@'
    ],
    [
        r"(.*?):(.*)",
        r"\g<1>\\:\g<2>:\g<1>",
    ],
    [
        r"child(.*)",
        "child\\g<1> > *",
    ],
]

usedClasses = findAllCSSClasses()
usedClasses = translateUsedClasses(usedClasses)

output = 'min.css'

newCSS = ''

for cssFile in cssFiles:
    # Parse the CSS File
    sheet = cssutils.parseFile(cssFile)
    rules = flattenStyleSheet(sheet)
    noClassSelectors = []
    for rule in rules:
        for usedClass in usedClasses:
            if '.' + usedClass == rule.selectorText:
                # If the class is used in the HTML, add it to the new CSS
                usedClasses.remove(usedClass) # Remove the class from the list
                if rule.parentRule:
                    newCSS += str(rule.parentRule.cssText)
                else:
                    newCSS += str(rule.cssText)
        if rule.selectorText[0] != '.' and not rule.selectorText in noClassSelectors: 
            # If the selector doesnt start with a dot and is not already in the list,
            # add it
            noClassSelectors.append(rule.selectorText)
            if rule.parentRule:
                newCSS += str(rule.parentRule.cssText)
            else:
                newCSS += str(rule.cssText)

newCSS = newCSS.replace('\n', '')
newCSS = newCSS.replace('  ', '')

with open(output, 'w') as f:
    f.write(newCSS)


print('TIME TOOK: ', time.time() - startTime)