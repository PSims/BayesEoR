// A class that holds parameters pertaining to various recognisers and
// a dialog to change them
//
// Copyright (c) 2000 Markus Demleitner <mdemleitner@head-cfa.harvard.edu>
//  This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
//
// tabsize=2

import java.awt.*;
import java.awt.event.*;
import java.util.*;


class intSlider extends Slider
{
	static final long serialVersionUID=20060308L;

	public intSlider(int min, int max, int init)
	{	
		super();
		this.setDispMode(SLIDERVALUE);
		this.setHeight(25);
		this.setSliderRange(min,max);
		this.setSliderValue(init);
	}
}


class DoubleSlider extends Slider
{
	static final long serialVersionUID=20060308L;

	public DoubleSlider(double min, double max, double init)
	{	
		super();
		this.setDispMode(SLIDERVALUE);
		this.setHeight(25);
		this.setSliderRange(min,max);
		this.setSliderValue(init);
	}

	protected String valString()
	{
		return Fmt.fmt(this.getSliderValue(), 3, 2);
	}
}


public class RecogniserSettings extends Frame
	implements ActionListener, SliderListener
{	
	Hashtable<String,Number> properties=new Hashtable<String,Number>();
	intSlider lineTracerSpacingField;
	DoubleSlider pointFinderThreshField;
	static final long serialVersionUID=20060308L;

	WindowAdapter winAd = new WindowAdapter()
	{	
		public void windowClosing(WindowEvent e)
		{
			close();
		}
	};


	public RecogniserSettings()
	{	
		this.setDefaults();
		this.makeDialog();
		this.setTitle("Dexter Recogniser Settings");
		this.setResizable(false);
	}


	private void setDefaults()
	{
		this.setIntProp("LineTracerSpacing",0);
		this.setDoubleProp("PointFinderThresh",0.15);
		this.setIntProp("blackThresh",110);
		this.setIntProp("weightThresh",50);
		if (this.isVisible())
			this.fillInValues();
	}


	public void setProp(String key, Number ob)
	{	
		this.properties.put(key, ob);
	}

	
	public void setIntProp(String key, int val)
	{
		this.setProp(key, new Integer(val));
	}


	public void setDoubleProp(String key, double val)
	{
		this.setProp(key, new Double(val));
	}


	public Object getProp(String key)
	{
		return this.properties.get(key);
	}


	public int getIntProp(String key)
	{
		return ((Integer)this.getProp(key)).intValue();
	}


	public double getDoubleProp(String key)
	{
		return ((Double)this.getProp(key)).doubleValue();
	}


	public void makeDialog()
	{	
		Panel settings = new Panel();
		GridBagLayout layout = new GridBagLayout();
		Font baseFont = new Font("Helvetica",Font.PLAIN,10);
		Font boldFont = new Font("Helvetica",Font.BOLD,10);

		this.setFont(baseFont);
		this.setLayout(new BorderLayout());
		settings.setLayout(layout);
		GridBagConstraints c = new GridBagConstraints();
		
		Label l=new Label("Line Tracer Settings");
		l.setFont(boldFont);
		c.gridy = 0;
		c.gridwidth = GridBagConstraints.REMAINDER;
		c.anchor = GridBagConstraints.WEST;
		layout.setConstraints(l,c);
		settings.add(l);

		l=new Label("Point Spacing (0 for automatic):");
		c.gridy = 1;
		c.gridwidth = 1;
		c.anchor = GridBagConstraints.WEST;
		layout.setConstraints(l,c);
		settings.add(l);

		lineTracerSpacingField = new intSlider(0,150,
			getIntProp("LineTracerSpacing"));
		lineTracerSpacingField.addSliderListener(this);
		c.gridwidth = GridBagConstraints.REMAINDER;
		c.anchor = GridBagConstraints.WEST;
		layout.setConstraints(lineTracerSpacingField,c);
		settings.add(lineTracerSpacingField);

		l=new Label("Point Finder Settings");
		l.setFont(boldFont);
		c.gridy = 2;
		c.gridwidth = GridBagConstraints.REMAINDER;
		c.anchor = GridBagConstraints.WEST;
		layout.setConstraints(l,c);
		settings.add(l);

		l=new Label("Distance Threshold:");
		c.gridy = 3;
		c.gridwidth = 1;
		c.anchor = GridBagConstraints.WEST;
		layout.setConstraints(l,c);
		settings.add(l);

		pointFinderThreshField = new DoubleSlider(0,0.66,
			getDoubleProp("PointFinderThresh"));
		pointFinderThreshField.addSliderListener(this);
		c.gridwidth = GridBagConstraints.REMAINDER;
		c.anchor = GridBagConstraints.WEST;
		layout.setConstraints(pointFinderThreshField,c);
		settings.add(pointFinderThreshField);

		settings.validate();
		add(settings,"Center");

		Panel buts = new Panel();
		buts.setLayout(new FlowLayout());
		Button defaultsButton = new Button("Defaults");
		buts.add(defaultsButton);
		defaultsButton.addActionListener(this);
		Button closeButton = new Button("Close");
		buts.add(closeButton);
		closeButton.addActionListener(this);

		add(buts,"South");

		pack();
	}


	public void fillInValues()
	{	
		this.lineTracerSpacingField.setSliderValue(
			getIntProp("LineTracerSpacing"));
		this.pointFinderThreshField.setSliderValue(
			getDoubleProp("PointFinderThresh"));
	}


	public void open()
	{
		if (this.isShowing())
			return;
		this.fillInValues();
		this.setVisible(true); 
		this.validate();
		this.addWindowListener(winAd);
	}


	public void apply()
	{
		this.setProp("LineTracerSpacing",new Integer(
			(int)Math.round(lineTracerSpacingField.getSliderValue())));
		this.setProp("PointFinderThresh",new Double(
			pointFinderThreshField.getSliderValue()));
	}


	public void close()
	{
		if (!this.isShowing())
			return;
		this.setVisible(false);
		this.removeWindowListener(winAd);
	}
	

	public void actionPerformed(ActionEvent e)
	{
		if (e.getActionCommand()=="Close")
			this.close();

		if (e.getActionCommand()=="Defaults") {	
			this.setDefaults();
			this.fillInValues();
		}

	}

	public void sliderStateChanged(SliderEvent e)
	{	
		this.apply();
	}

}
