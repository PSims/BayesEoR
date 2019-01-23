// The main object when run as an application
//
// Copyright (c) 2003 Markus Demleitner <msdemlei@cl.uni-heidelberg.de>
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
import java.awt.image.*;

class Debuxter_w extends Frame implements MainServices
{	
	DExtractor worker;
	ScrollImScrollBar sel;
	DExtractor window = null;
	String sourcebib,sourcepage;
	boolean onIndexedDisplay=false;
	ImageGetter imageGetter;
	DataDeliverer dataDeliverer = new PlainDataDeliverer(this);
	static final long serialVersionUID=20060308L;

	public Debuxter_w(String args[])
	{ 
		Rectangle bbox=new Rectangle();
		ColorModel colormodel = Toolkit.getDefaultToolkit().getColorModel();

		if (colormodel instanceof IndexColorModel) {
			onIndexedDisplay = true;
		}

		imageGetter = new PlainImageGetter(this, args[0]);
		sourcebib = args[0];
		sourcepage = "unknown";
		bbox.x = 0;
		bbox.y = 0;
		bbox.width = 200;
		bbox.height = 200;
		window = new DExtractor(this, bbox, sourcebib, sourcepage, imageGetter,
			dataDeliverer);
		window.setTitle("Dexter");
		window.pack();
		window.setVisible(true);
	}

	public void showHelp()
	{	
		System.out.println("No help in Debuxter");
	}

	public void notifySelection(Rectangle bbox)
	{	
		System.out.println("Debuxter doesn't use this.");
	}

	public void childClosed()
	{
		Runtime.getRuntime().exit(0);
	}

	public void notifyChangedSize()
	{
	}
}

public class Debuxter
{
	public static void main(String[] args)
	{
		new Debuxter_w(args);
	}
}
