// This class is used by dexter to acquire images
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
//tabsize=2

import java.applet.*;
import java.awt.Image;
import java.awt.Rectangle;
import java.net.URL;
import java.net.MalformedURLException;

public class AppletImageGetter implements ImageGetter
{	
	Applet parent;
	String sourceURL;

	public AppletImageGetter(Applet parent,String sourceURL)
	{
		this.parent = parent;
		this.sourceURL = sourceURL;
	}

	public Image getImage(int scale)
	{	
		try {	
			return parent.getImage(new URL(sourceURL+"&scale="+scale));
		} catch (MalformedURLException e) {
			System.out.println("Could not get full page image"); 
		}
		return null;
	}

	public Image getImage(int scale, Rectangle bbox) 
	{ 
		String targurl;

		if (bbox==null) {
			return this.getImage(scale);
		}
		targurl = sourceURL+"&scale="+scale+"&coord="+bbox.x+","+bbox.y+","+
			(bbox.x+bbox.width)+","+(bbox.y+bbox.height);
		try { 
			return this.parent.getImage(new URL(targurl));
		} catch (MalformedURLException e) {	
			System.out.println("Could not get full page image"); 
		}
		return null;
	}
}

// vi:ts=2:
