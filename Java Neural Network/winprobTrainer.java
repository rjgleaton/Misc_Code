/* RJ Gleaton 2020
*  This is an /attempt/ at a neural network meant to semi-accurately predict the win probability of two players
*  at any given frame within a match on AikidoBigDojo.tbm in the game Toribash
*  Sorry if the code is a bit sloppy, this was my first attempt so there's been a lot of trial and error
*  Coded in Java due to Python possibly being too slow, though in hindsight it should have been done in C++
*  Sort of reaches a local min, it always give player 2 (Uke) a 55% chance of winning, upon analysis player 2
*  wins 55% of the time in the matches provided, the inputs probably need to be lessened to just a few key
*  variables to make the results easier to predict. Also the training data was gathered from public lobbies where
*  the outcome might not always be the same and is subject to, for lack of a better phrase, the stupidity of the players.
*/
import com.sun.javafx.image.IntPixelGetter;

import java.io.*;
import java.util.*;

public class winprobTrainer {

    public static final String DELIM = " ";
    public static final File folder = new File("C:/Program Files (x86)/Steam/steamapps/common/Toribash/replay/autosave"); //Location of the training data
    public static final int NUMVARIABLES = 632; //Number of inputs from the file for any given freeze frame of a replay
    public static final int layer1Nodes = 32; 
    public static final int layer2Nodes = 16;
    public static double weights[][] = new double[NUMVARIABLES][layer1Nodes]; //weights of input layer
    public static double weights2[][] = new double[layer1Nodes][layer2Nodes]; //weights of hidden layer
    public static double weights3[] = new double[layer2Nodes]; //weights for hidden layer 2 going to output
    public static double weightsChange[][] = new double[NUMVARIABLES][layer1Nodes]; //stores average change data for each weight
    public static double weights2Change[][] = new double[layer1Nodes][layer2Nodes];
    public static double weights3Change[] = new double[layer2Nodes];


    public static double sigmoid(double input){
        return 1/(1+Math.pow(Math.E,-1*input));
    }
    //sigmoid derivative. I've tried several different variantions for both of these
    public static double sigdiv(double input){
        double e = Math.E;
        return Math.pow(e,-1*input)/Math.pow(1+Math.pow(e,-1*input),2);
    }


    public static double feedforward(int winner, int frame, int[][] fracture, int[][] dismember, int[][] grip, double[][] position, double[][] linVelo, double[][] angVelo, double[][] quat, int pointDif, int x){
        ArrayList<Double> allinone = new ArrayList<Double>(); //Store all the data in one array list just for simplicity
        //double[] layer1 = new double[layer1Nodes];
        allinone.add((double)frame); //frames # at 0
        for(int j = 0; j < 2; j++){
            for(int k = 0; k < 20; k++){
                allinone.add((double)fracture[j][k]);
            }
        }
        for(int j = 0; j < 2; j++){
            for(int k = 0; k < 20; k++){
                allinone.add((double)dismember[j][k]);
            }
        }
        for(int j = 0; j < 2; j++){
            for(int k = 0; k < 2; k++){
                allinone.add((double)grip[j][k]);
            }
        }
        for(int j = 0; j < 2; j++){
            for(int k = 0; k < 63; k++){
                allinone.add(position[j][k]);
            }
        }
        for(int j = 0; j < 2; j++){
            for(int k = 0; k < 63; k++){
                allinone.add(linVelo[j][k]);
            }
        }
        for(int j = 0; j < 2; j++){
            for(int k = 0; k < 63; k++){
                allinone.add(angVelo[j][k]);
            }
        }
        for(int j = 0; j < 2; j++){
            for(int k = 0; k < 84; k++){
                allinone.add(quat[j][k]);
            }
        }
        allinone.add((double)pointDif/1000);


        //load layer1 with values and 'sigmoid' them
        double layer1[] = new double[layer1Nodes];
        for(int i = 0; i < layer1.length; i++){
            layer1[i] = 0;
        }
        for(int i = 0; i < layer1.length; i++){
            for(int j = 0; j < allinone.size(); j++){
                layer1[i] += allinone.get(j)*weights[j][i];
            }
            layer1[i] = sigmoid(layer1[i]);
        }

        //load layer2 with values and sigmoid
        double layer2[] = new double[layer2Nodes];
        for(int i = 0; i < layer2.length; i++){
            layer2[i] = 0;
        }
        for(int i = 0; i < layer2.length; i++){
            for(int j = 0; j < layer1.length; j++){
                layer2[i] += layer1[j]*weights2[j][i];
            }
            layer2[i] = sigmoid(layer2[i]);
        }

        //get output
        double output = 0;
        for(int i = 0; i < weights3.length; i++){
            output += layer2[i]*weights3[i];
        }
        output = sigmoid(output); //output is put through sigmoid, i've tried other methods but this works the best

        double error = winner - output;
        double cost = winner - output;
        
        //propegate backwards
        propBack(error, allinone, layer1, layer2, output, winner);
        //double cost = 0.5*Math.pow(winner - output,2);

        //Debug code, ignore
        /** 
        if(x == 999 || x ==0 || x == 1 || x ==3){
            print("Frame "+frame);
            print("Output: "+output);
            print("Winner: "+winner);
            print("Error: "+error);
            print("X = "+x);
        }
        */
        
        return cost;
    }

    public static void propBack(double error, ArrayList<Double> input, double[] layer1, double[] layer2, double output, int winner){
        /**
        //using delta rule
        //back prop to hidden layer 2 from output
        for(int i = 0; i < layer2.length; i++){
            weights3Change[i] += 0.1*(-1*error)*output*layer2[i]*(1-output);
        }

        //back prop to hidden layer 1 from layer 2
        for(int i = 0; i < layer1.length; i++){
            for(int j = 0; j < layer2.length; j++){
                weights2Change[i][j] += 0.1*(-1*error)*layer1[i]*layer2[j]*(1-layer2[j]);
            }
        }


        //back prop to input from hidden
        for(int i = 0; i < input.size(); i++){
            for(int j = 0; j < layer1.length; j++){
                weightsChange[i][j] += 0.1*(-1*error)*input.get(i)*layer1[j]*(1-layer1[j]);
            }
        }
         **/

        for(int i = 0; i < layer2.length; i++){
            weights3Change[i] += 0.01*(error*(output*(1-output))*(layer2[i]));
            weights3[i] += weights3Change[i];
        }
        double[] weights2ChangeHolder = new double[layer1.length];
        for(int i = 0; i < layer1.length; i++){
            for(int j = 0; j < layer2.length; j++){
                weights2Change[i][j] += 10.01*(weights3Change[j]*(layer2[j]*(1-layer2[j]))*layer1[i]);
                weights2[i][j] += weights2Change[i][j];
                weights2ChangeHolder[i] += weights2Change[i][j];
            }
        }
        for(int i = 0; i < input.size(); i++){
            for(int j = 0; j < layer1.length; j++){
                weightsChange[i][j] += 0.01*(weights2ChangeHolder[j]*(layer1[j]*(1-layer1[j]))*input.get(i));
                weights[i][j] += weightsChange[i][j];
            }
        }
    }



    public static void main(String[] args){

        //First calculate weights
        for(int i = 0; i < NUMVARIABLES; i++){
            for(int j = 0; j < layer1Nodes; j++){
                weights[i][j] = Math.random();
            }
        }
        for(int i = 0; i < layer1Nodes; i++){
            for(int j = 0; j < layer2Nodes; j++){
                weights2[i][j] = Math.random();
            }
        }
        for (int j = 0; j < weights3.length; j++) {
            weights3[j] = Math.random()*0.09;
        }

        //Folder debugging, ignore
        if(folder.canRead())
            print("Can read");
        else
            print("can't read");

        if(folder.isDirectory())
            print("is Directory");
        else
            print("is not directory");
        File[] listOfFiles = folder.listFiles();
        /**
        for (File file : listOfFiles) {
            if (file.isFile()) {
                System.out.println(file.getName());
            }
        }
        print("Number of files = " + listOfFiles.length);
        print("First file = " + listOfFiles[0]);
         **/

        //amount of time the program runs
        //calculates the average change for all weights with the given replays before propegating backwards
        for (int x = 0; x < 1000; x++) {

            double avgCost = 0;
            int counter = 0; //counts number of time feed forward is used

            //set avg change to be 0
            for(int i = 0; i<NUMVARIABLES; i++){
                for(int j = 0; j<layer1Nodes; j++){
                    weightsChange[i][j] = 0;
                }
            }
            for(int i = 0; i<layer1Nodes; i++){
                for(int j = 0; j<layer2Nodes; j++){
                    weights2Change[i][j] = 0;
                }
            }
            for(int i = 0; i<layer2Nodes; i++){
                weights3Change[i] = 0;
            }

            try {
                for (int i = 0; i < listOfFiles.length; i++) {
                    File fileName = listOfFiles[i];

                    //preparing all the inputs 
                    int winner;
                    int frame = 0;
                    int fracture[][] = new int[2][20];
                    int dismember[][] = new int[2][20];
                    int grip[][] = new int[2][2];
                    for (int j = 0; j < 20; j++) { //set all joints fractures and dms equal to 0
                        fracture[0][j] = 0;
                        fracture[1][j] = 0;
                        dismember[0][j] = 0;
                        dismember[1][j] = 0;
                    }
                    double position[][] = new double[2][63];
                    double linVelo[][] = new double[2][63];
                    double angVelo[][] = new double[2][63];
                    double quat[][] = new double[2][84];
                    int pointDif = 0;

                    Scanner getData = new Scanner(new File("C:/Program Files (x86)/Steam/steamapps/common/Toribash/replay/autosave/" + fileName.getName()));

                    //inserting all the data into the arrays
                    for (int j = 0; j < 3; j++) {
                        getData.nextLine();
                    } //Get winner
                    String winnerLine = getData.nextLine();
                    String[] winnerInfo = winnerLine.split(DELIM);
                    //winnerLine = winnerInfo[2];
                    getData.nextLine();
                    getData.nextLine();
                    String[] winnerName = getData.nextLine().split(DELIM);
                    //print(fileName.getName());
                    if (winnerInfo[2].equals(winnerName[2])) winner = 0; //0 for tori
                    else winner = 1; // 1 for uke

                    while (getData.hasNextLine()) {
                        String fileLine = getData.nextLine();
                        String[] lineData = fileLine.split(DELIM);

                        if (lineData[0].equalsIgnoreCase("frame")) { //found frame data
                            lineData[1] = lineData[1].substring(0, lineData[1].length() - 1); // remove semicolon
                            frame = Integer.parseInt(lineData[1]); //set frame data
                            pointDif = Integer.parseInt(lineData[2]) - Integer.parseInt(lineData[3]);
                            //print("Frame: "+Integer.toString(frame));

                            if(frame != 0) {
                                avgCost += Math.abs(feedforward(winner, frame, fracture, dismember, grip, position, linVelo, angVelo, quat, pointDif, x));
                                counter++;
                                //print("Sigmoid: " + finalAns);
                                //print("Did uke win: " + winner);
                            }
                        }

                        if (lineData[0].equalsIgnoreCase("grip")) {
                            lineData[1] = lineData[1].substring(0, lineData[1].length() - 1);
                            grip[Integer.parseInt(lineData[1])][0] = Integer.parseInt(lineData[2]);
                            grip[Integer.parseInt(lineData[1])][1] = Integer.parseInt(lineData[3]);
                            //print("Grip: "+ grip[0][1]+" "+grip[0][0]);
                        }

                        if (lineData[0].equalsIgnoreCase("pos")) {
                            lineData[1] = lineData[1].substring(0, lineData[1].length() - 1);
                            for (int j = 0; j < 63; j++) {
                                position[Integer.parseInt(lineData[1])][j] = Double.parseDouble(lineData[j + 2]);
                                //print("Pos: "+Double.parseDouble(lineData[j + 2]));
                            }
                            //print(Double.toString(position[0][61]));
                        }

                        if (lineData[0].equalsIgnoreCase("qat")) {
                            lineData[1] = lineData[1].substring(0, lineData[1].length() - 1);
                            for (int j = 0; j < 84; j++) {
                                quat[Integer.parseInt(lineData[1])][j] = Double.parseDouble(lineData[j + 2]);
                            }
                        }

                        if (lineData[0].equalsIgnoreCase("linvel")) {
                            lineData[1] = lineData[1].substring(0, lineData[1].length() - 1);
                            for (int j = 0; j < 63; j++) {
                                linVelo[Integer.parseInt(lineData[1])][j] = Double.parseDouble(lineData[j + 2]);
                            }
                        }

                        if (lineData[0].equalsIgnoreCase("angvel")) {
                            lineData[1] = lineData[1].substring(0, lineData[1].length() - 1);
                            for (int j = 0; j < 63; j++) {
                                angVelo[Integer.parseInt(lineData[1])][j] = Double.parseDouble(lineData[j + 2]);
                            }
                        }

                        if (lineData[0].equalsIgnoreCase("crush")) {
                            lineData[1] = lineData[1].substring(0, lineData[1].length() - 1);
                            for (int j = 0; j < lineData.length - 2; j++) {
                                linVelo[Integer.parseInt(lineData[1])][Integer.parseInt(lineData[j + 2])] = 1;
                            }
                        }

                        if (lineData[0].equalsIgnoreCase("fract")) {
                            lineData[1] = lineData[1].substring(0, lineData[1].length() - 1);
                            for (int j = 0; j < lineData.length - 2; j++) {
                                linVelo[Integer.parseInt(lineData[1])][Integer.parseInt(lineData[j + 2])] = 1;
                            }
                        }

                    }
                    getData.close();
                }
            } catch (FileNotFoundException e) {
                System.out.println("File not found");
            } catch (NoSuchElementException e) {
                System.out.println("Element Error");
            }

            /**
            for(int i = 0; i<NUMVARIABLES; i++){
                for(int j = 0; j<layer1Nodes; j++){
                    weights[i][j] = weightsChange[i][j];
                }
            }
            for(int i = 0; i<layer1Nodes; i++){
                for(int j = 0; j<layer2Nodes; j++){
                   weights2[i][j] = weights2Change[i][j];
                }
            }
            for(int i = 0; i<layer2Nodes; i++){
                weights3[i] = weights3Change[i];
            }
            **/
            avgCost = avgCost/counter;
            print(""+avgCost);
        }

        try{
            FileWriter out = new FileWriter("output.txt");
            out.write("Input weights: ");
            for(int i = 0; i<NUMVARIABLES; i++){
                for(int j = 0; j<layer1Nodes; j++){
                    out.write(weights[i][j] + " ");
                }
            }
            out.write("\n");
            out.write("Layer 1 weights: ");
            for(int i = 0; i<layer1Nodes; i++){
                for(int j = 0; j<layer2Nodes; j++){
                    out.write(weights2[i][j] + " ");
                }
            }
            out.write("\n");
            out.write("Layer 2 weights: ");
            for(int i = 0; i<layer2Nodes; i++){
                out.write(weights3[i] + " ");
            }
            out.close();
        } catch (NoSuchElementException e) {
            System.out.println("Element Error");
        } catch (IOException e) {
            System.out.println("An error occurred.");
        }

    }

    public static void print(String toPrint){
        System.out.println(toPrint);
    }
}
