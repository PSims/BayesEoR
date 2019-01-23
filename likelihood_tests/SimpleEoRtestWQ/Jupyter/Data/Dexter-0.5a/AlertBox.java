// A hacked class to display an alert box
// Why exactly doesn't awt provide this class?
//
// Copyright (c) 2000 Markus Demleitner
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

class AlertBox extends Dialog
	implements ActionListener
{
	static final long serialVersionUID=20060308L;

	AlertBox(Frame parent, String title, String message)
	{	
		super(parent, title, true);
		this.buildFrame(title, message);

		// try to get the alert box centered on parent widget
		Point p = parent.getLocation();
    Dimension d = parent.getSize();
    Dimension s = this.getSize();
    p.translate((d.width-s.width)/2, (d.height-s.height)/2);
    this.setLocation(p);
		this.setVisible(true);
	}

	protected void buildFrame(String title, String message)
	{
		final char linesep = '\n';

		this.setFont(new Font("Helvetica", Font.PLAIN, 14));

		GridBagLayout layout = new GridBagLayout();
		GridBagConstraints c = new GridBagConstraints();
		this.setLayout(layout);

		c.fill = GridBagConstraints.NONE;
		c.gridwidth = GridBagConstraints.REMAINDER;
		c.weightx = 1;
		c.weighty = 1;
		c.anchor = GridBagConstraints.CENTER;

		Button but=new Button("Ok");
		but.addActionListener(this);
		
		int lpos, curpos=0;
		// Add lines in message line by line in seperate labels
		while (-1!=(lpos=message.indexOf(linesep, curpos))) { 
			Label lab=new Label(message.substring(curpos, lpos));
			layout.setConstraints(lab, c);
			add(lab);
			curpos = lpos+1;
		}
		// handle the last line -- gee, I wish for str.split()
		Label lab = new Label(message.substring(curpos, message.length()));
		layout.setConstraints(lab, c);
		this.add(lab);

		// add the button
		c.ipady = 10;
		c.ipadx = 10;
		layout.setConstraints(but, c);
		this.add(but);
		this.pack();
	}

	public void actionPerformed(ActionEvent evt)
	{	
		// Button pressed, withdraw window
		this.setVisible(false);
		this.dispose();
	}
}

// vi:ts=2:
