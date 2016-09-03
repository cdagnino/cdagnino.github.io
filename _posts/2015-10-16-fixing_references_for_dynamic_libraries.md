---
layout: post
title:  "Fixing references for dynamic libraries in R and Python"
date:   2015-10-16 18:14:50
categories: computing
image: /assets/article_images/rstudio_crazy.png
---


Every once in a while, I'll import an `R` or `Python` library and I'll get an error telling me that it can't find some scary fortran library. This happened in `R`:

{% highlight r %}
library(lme4)
Error in dyn.load(file, DLLpath = DLLpath, ...) : 
  unable to load shared object '/Users/your_username/Rlibs/minqa/libs/minqa.so':
  dlopen(/Users/your_username/Rlibs/minqa/libs/minqa.so, 6):
   Library not loaded: /usr/local/lib/gcc/4.9/libgfortran.3.dylib
  Referenced from: /Users/your_username/Rlibs/minqa/libs/minqa.so
  Reason: image not found
{% endhighlight %}

If I really needed that library I would either cry and/or try to uninstall and install everything again, hoping things would get solved. I eventually learned the proper fix, so I though I'd share it with you. I use a Mac, but I'm guessing very similar (if not the same) steps apply for Linux.

The usual problem is that you have the dynamic library (`dylib`) but the compiled `.so` file is looking at it in the wrong place.

For this example, let's look at `minqa.so`, which is trying to find `libgfortran.3.dylib` in `gcc/4.9/`.

To check the references of `minqa.so`, we run on the terminal:

{% highlight bash %}
otool -L minqa.so
{% endhighlight %}

which printed:

{% highlight bash %}
minqa.so:
	minqa.so (compatibility version 0.0.0, current version 0.0.0)
	/usr/local/lib/gcc/4.9/libgfortran.3.dylib (compatibility version 4.0.0, current version 4.0.0)
	/usr/local/lib/gcc/4.9/libquadmath.0.dylib (compatibility version 1.0.0, current version 1.0.0)
{% endhighlight %}

Ok. I looked at my `lib/gcc` folder and realized I had version 5, not 4.9. The `libgfortran3.dylib` was there in the 5 folder. How can I communicate the good news to `minqa.so`?

{% highlight bash %}
install_name_tool -change /usr/local/lib/gcc/4.9/libgfortran.3.dylib
   /usr/local/lib/gcc/5/libgfortran.3.dylib minqa.so
{% endhighlight %}

You might need to put `sudo` before that command.
In general the syntax is

{% highlight bash %}
install_name_tool -change old_location new_location file_you_wanna_change.so
{% endhighlight %}

If you don't know where the `dylib` library is, you can search for it with Spotlight or with `locate` on the command line.

Also, sometimes the `dylib` won't even have a location and the `otool` step will output only `libgfortran.3.dylib`. The same steps above apply, but the `old_location` will just be the name of the dynamic library.

You can read more about this issue [in this stackoverflow question](http://stackoverflow.com/questions/6383310/python-mysqldb-library-not-loaded-libmysqlclient-18-dylib)