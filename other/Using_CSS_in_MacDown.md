## Using CSS in MacDown

If you need to fine tune the appearance of a document, you can always insert some CSS code ([read more about CSS](http://www.w3schools.com/css/css_intro.asp)).

At the beginning of your MacDown file, write:

```css
<style>
...your CSS code here...
</style>
```

For example, if you want to modify the font, size and background color of the "body" segment of your document, you can write

```css
<style>body {
font-family: "Avenir Next", Helvetica, Arial, sans-serif;
background:#fefefe;
font-size: 20px;
}
</style>
```


### Show the entire code block

To avoid vertical clipping of a long code block, you can use

```css
<style>
pre[class*="language-"] {
max-height: none;
}
</style>
```


### Avoid horizontal wrap

To avoid horizontal wrap of your code block:

```css
<style>
code, pre {
white-space: pre-wrap !important;
}
</style>
```

To find out more, you can click Reveal in the Rendering pane of the preferences to get the CSS files of the different rendering styles in MacDown.

[This page](https://github.com/dac09/Macdown-styles) has further examples of CSS for MacDown.



## Writing comments

A way to write comment (so it doesn't get rendered) is:



```
<!---
your comment goes here
and here
-->
```


