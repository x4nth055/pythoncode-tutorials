# Minify CSS with Python
**Learn how to utilize cssutils to minimize CSS files in a Web Project**


## Idea

In this article, we will make a python program that will search for classes used in all HTML files in a project and it will then search and compile these files from the CSS files. The program will serve a specific purpose as it will match classes strictly; which means `bg-black` won't `bg-black:hover`, The used classes have to appear in the stylesheets as they are used. This way of minimizing is useful for utility classes such as `width-800px` or `color-grey-800` that only change on the property. Now maybe your utility classes also entail something like this pattern: `child-margin-2rem` which in the stylesheet is actually `child-margin-2rem > *`, this won't match by default but we will make it possible to replace such patterns with the appropriate selector. Finally, you can change the code so the minified works better for your case or you could even redo it on your own with the knowledge gained.

We will utilize a CSS Library called CSSUtils that allows us to parse, read and write CSS.

## Imports

Let's start with the Modules and Libraries we have to import for our little program. The most important will be `cssutils` which has to be downloaded with `pip install cssutils`. We also want to import `re`, `os`, `time`. We get the logging module simply to turn off logging because cssutils throws a lot of errors. We then clear the console with `os.system` and we save the start time of the program to a variable.

```py
import cssutils
import re
import logging
import os
import time
cssutils.log.setLevel(logging.CRITICAL)

startTime = time.time()
os.system('cls')
```

## Getting the Files

Firstly we get lists of files ending in `.html` and `.css`.  We save these lists for later. 

```py
htmlFiles = getFilesByExtension('.html', '.')

cssFiles = getFilesByExtension('.css', 'style')
```

Let's also go over the function that searches for all these files. keep in mind it has to be defined before its usage. Here we use the `walk` function of `os` which receives a path and it will return data about each subdirectory and the directory itself. We only need the files which are the third item of the returned tuple. We loop over these and if they end with the specified extension we add them to the `foundFiles` list. Lastly, we also need to return this list. 

```py
def getFilesByExtension(ext, root):
    foundFiles = []

    for root, directories, files in os.walk(root):

        for f in files:

            if f.endswith(ext):
                foundFiles.append(os.path.join(root, f))
```

## Finding all used Classes

Next up we want to find all used classes in all HTML files that were found. To do this we first create a dictionary to store each class name as an item. We do it this way so we don't have duplicates in the end. We then loop over all HTML files and for each one we get the content and we use a Regular Expression to find all class strings. Continuing we split each of these found strings because classes are separated by a space. Lastly, we return the found list dictionary but we return the keys which are the classes.

```py
usedClasses = findAllCSSClasses()

# Function, defined before
def findAllCSSClasses():
    usedClasses = {}

    # Find all used classes
    for htmlFile in htmlFiles:
        with open(htmlFile, 'r') as f:
            htmlContent = f.read()

        regex = r'class="(.*?)"'

        matched = re.finditer(regex, htmlContent, re.MULTILINE | re.DOTALL)

        for i in matched:

            for className in i.groups()[0].split(' '):
                usedClasses[className] = ''

    return list(usedClasses.keys())
```

## Translating used Classes

Now wer translate some classes, this is useful if the class name won't exactly match the selector, but it follows a pattern like all classes starting with `child-` have `> *` appended to their selector, and here we handle this. We define each translation in a list where the first item is the regex and the second is the replacement.

```py
# Use Translations if the class names in the Markup don't exactly 
# match the CSS Selector ( Except for the dot at the beginning. )
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

usedClasses = translateUsedClasses(usedClasses)
```

In the function we then loop over each regex for each class, so every translation is potentially applied to each class name. We then simply apply the replacement with the `re.sub` method.

```py
def translateUsedClasses(classList):

    for i, usedClass in enumerate(classList):
        for translation in translations:

            regex = translation[0]
            subst = translation[1]

            if re.search(regex, usedClass):
                result = re.sub(regex, subst, usedClass, 1, re.MULTILINE)

                classList[i] = result

    return classList
```

## Getting used Classes from the Stylesheets

After that, we get the style definition from the stylesheets with cssutils, before we loop over the found style sheets we first define the path of the minified CSS which in this case is `min.css` then we also create a variable called `newCSS` that will hold the new CSS content.

```py
output = 'min.css'

newCSS = ''
```

We continue by looping over all CSS files. We parse each file with `cssutils.parsefile(path)` and get all the rules in the style sheet with the custom `flattenStyleSheet()` function, we later go over how it works but it will essentially put all rules hidden inside media queries into the same list as top-level rules. then we define a list that will hold all selector names that are not classes that we encounter. We do this because something like `input` should not be left out. Then we loop over each rule and each class and if the selector and selector text of the rule match up we add the whole CSS text of the rule to the newCSS string. We simply need to watch out if the rule has a parent rule which would be a media query. We do the same thing for all the rules not starting with a dot.

```py
for cssFile in cssFiles:

    sheet = cssutils.parseFile(cssFile)
    rules = flattenStyleSheet(sheet)

    noClassSelectors = []

    for rule in rules:
        for usedClass in usedClasses:

            if '.' + usedClass == rule.selectorText:
                usedClasses.remove(usedClass)

                if rule.parentRule:
                    newCSS += str(rule.parentRule.cssText)
                else:
                    newCSS += str(rule.cssText)

        if rule.selectorText[0] != '.' and not rule.selectorText in noClassSelectors:

            noClassSelectors.append(rule.selectorText)

            if rule.parentRule:
                newCSS += str(rule.parentRule.cssText)
            else:
                newCSS += str(rule.cssText)
```

### `flattenstylesheet` function

Lets quickly go over the flattenstylesheet function. It will receive the sheet and it loops over each rule in that sheet in, then it will check if the rule is simply a style rule or media rule so it can add all rules to a one-dimensional list.

```py
def flattenStyleSheet(sheet):
    ruleList = []

    for rule in sheet.cssRules:

        if rule.typeString == 'MEDIA_RULE':
            ruleList += rule.cssRules

        elif rule.typeString == 'STYLE_RULE':
            ruleList.append(rule)
    
    return ruleList
```

## Saving new CSS

Lastly, we minify the CSS further by removing linebreaks and double spaces and we save this new CSS to the specified location. 

```py
newCSS = newCSS.replace('\n', '')
newCSS = newCSS.replace('  ', '')

with open(output, 'w') as f:
    f.write(newCSS)


print('TIME: ', time.time() - startTime)
```

## Conclusion

Excellent! You have successfully created a CSS Minifier using Python code! See how you can add more features to this program such as a config file for further options. Also keep in mind that this program could need some optimization since it runs very slow on larger projects.