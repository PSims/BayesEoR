.SUFFIXES:
	.class .java

JAVAOPTS=-Xlint

%.class:%.java
	javac $(JAVAOPTS) $<

RELEASE=0.5a

ALLJAVA=Dexter.java DExtractor.java ImageWithPoints.java ScrollImScrollBar.java\
	ImageGetter.java DataDeliverer.java Fmt.java AppletImageGetter.java\
	AppletDataDeliverer.java MainServices.java Recogniser.java\
	LineTracer.java MagGlass.java AlertBox.java PointFinder.java\
	AutoAxisFinder.java RecogniserSettings.java Slider.java\
	SliderListener.java Datapoint.java AffineTrafo.java \
	CantComputeException.java Gauge.java MissingData.java \
	DoublePoint.java Comparator.java QuickSort.java Sortable.java
ALLCLASSES=$(subst .java,.class,$(ALLJAVA))
DEBJAVA=Debuxter.java PlainDataDeliverer.java PlainImageGetter.java
DEBCLASSES=$(subst .java,.class,$(DEBJAVA))
GAUCHOJAVA=Gaucho.java ScriptImageGetter.java PlainDataDeliverer.java
GAUCHOCLASSES=$(subst .java,.class,$(GAUCHOJAVA))
OTHERDISTRI=Dexterhelp.html COPYING COPYING.ncsa README Makefile\
	HOWTO.standalone Changelog runGaucho dexter_getImage example.gif\
	sample.ps Gaucho.manifest Debuxter.manifest
ADSPERLFILES=ads/dexterstart.pl ads/dp_receive.pl ads/dp_send.pl

debug: debuxter.jar
	java -mx30000000 -jar Debuxter.jar example.gif

debuxter.jar: $(ALLCLASSES) $(DEBCLASSES)
	jar cfm Debuxter.jar Debuxter.manifest *.class

Dexter.jar: $(ALLCLASSES) $(ALLPERLFILES)
	jar -cf Dexter.jar *.class

Gaucho.jar: $(ALLCLASSES) $(GAUCHOCLASSES)
	jar cfm Gaucho.jar Gaucho.manifest *.class

test: Gaucho.jar
	./runGaucho sample.ps

ptest: Gaucho.jar
	./runGaucho example.gif

clean:
	rm -f *.jar *.class

local: Dexter.jar
	cp Dexter.jar /var/lib/httpd/htdocs/Dexter

DESTDIR=Dexter-$(RELEASE)

distri:
	if test ! -d $(DESTDIR); then mkdir $(DESTDIR); fi
	cp $(ALLJAVA) $(DEBJAVA) $(GAUCHOJAVA) $(DESTDIR)
	cp $(OTHERDISTRI) $(DESTDIR)
	cp example.gif $(DESTDIR)
	tar -cvf $(DESTDIR).tar $(DESTDIR)
	-rm $(DESTDIR).tar.gz
	gzip -9 $(DESTDIR).tar

adsinstall: $(ALLCLASSES) $(ADSPERLFILES)
	jar -cf Dexter.jar *.class
	install -g ads -m 664 Dexter.jar $$DOCUMENT_ROOT/Dexter
	install -g ads -m 775 $(ALLPERLFILES) $(ADSCGIBIN)/noarch

adstestinstall: $(ALLCLASSES)  $(ADSPERLFILES)
	jar -cf Dexter.jar *.class
	install -g ads -m 664 Dexter.jar $$DOCUMENT_ROOT/install/Dexter
	install -g ads -m 775 $(ADSPERLFILES) $(ADSCGIBIN)/install/noarch

