// A Data Deliverer for debug purposes
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

import java.io.*;

class PlainDataDeliverer implements DataDeliverer
{	Object parent;

	public PlainDataDeliverer(Object parent)
	{
		this.parent = parent;
	}

	public void deliver(String toSend, String fname, int option)
	{	
		try {
			FileWriter outFile = new FileWriter(fname);
			outFile.write(toSend);
			outFile.close();
		} catch (IOException whatever) {
			System.out.println("Sorry, couldn't save.\n");
		}
	}
}
