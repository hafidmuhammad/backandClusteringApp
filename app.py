from flask import Flask, jsonify, request, make_response
from flask_restful import Api, Resource, reqparse
from flask_pymongo import PyMongo
from bcrypt import hashpw, gensalt, checkpw
from bson.objectid import ObjectId
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/clusteringapp"
mongo = PyMongo(app)
api = Api(app)

class UserList(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("firstName", type=str, required=True)
        parser.add_argument("lastName", type=str, required=True)
        parser.add_argument("phoneNumber", type=str, required=True)
        parser.add_argument("address", type=str, required=True)
        parser.add_argument("email", type=str, required=True)
        parser.add_argument("password", type=str, required=True)
        args = parser.parse_args()

        password_hash = hashpw(args["password"].encode("utf-8"), gensalt())

        new_user = {
            "firstName": args["firstName"],
            "lastName": args["lastName"],
            "phoneNumber": args["phoneNumber"],
            "address": args["address"],
            "email": args["email"],
            "password": password_hash,
        }

        result = mongo.db.users.insert_one(new_user)
        return {"message": "User added successfully", "user_id": str(result.inserted_id)}

class UserLogin(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("email", type=str, required=True)
        parser.add_argument("password", type=str, required=True)
        args = parser.parse_args()

        user = mongo.db.users.find_one({"email": args["email"]})

        if user and checkpw(args["password"].encode("utf-8"), user["password"]):
            # Password is correct
            response = make_response({"message": "Login successful"})
            # You can set JWT token or session here
            return response
        else:
            return {"message": "Invalid login credentials"}, 401

class FinancialReport(Resource):
    def post(self):
        data = request.get_json()
        stockname = data.get("stockname", "")
        roa = float(data.get("roa"))
        roe = float(data.get("roe"))
        eps = float(data.get("eps"))
        npm = float(data.get("npm"))
        bv = float(data.get("bv"))
        price_to_bv = float(data.get("priceToBv"))
        pe_ratio = float(data.get("peRatio"))
        de_ratio = float(data.get("deRatio"))

        report = {
            "stockname": stockname,
            "roa": roa,
            "roe": roe,
            "eps": eps,
            "npm": npm,
            "bv": bv,
            "price_to_bv": price_to_bv,
            "pe_ratio": pe_ratio,
            "de_ratio": de_ratio
        }

        mongo.db.financial_reports.insert_one(report)
        return {"message": "Financial report added successfully"}

class FinancialReportList(Resource):
    def get(self):
        reports = list(mongo.db.financial_reports.find())
        return {"reports": reports}

class FinancialReportDetail(Resource):
    def get(self, report_id):
        report = mongo.db.financial_reports.find_one({"_id": ObjectId(report_id)})
        if report:
            return {"report": report}
        return {"message": "Report not found"}, 404

    def put(self, report_id):
        data = request.get_json()
        updated_report = {
            "stockname": data.get("stockname", ""),
            "roa": float(data.get("roa")),
            "roe": float(data.get("roe")),
            "eps": float(data.get("eps")),
            "npm": float(data.get("npm")),
            "bv": float(data.get("bv")),
            "price_to_bv": float(data.get("priceToBv")),
            "pe_ratio": float(data.get("peRatio")),
            "de_ratio": float(data.get("deRatio"))
        }
        result = mongo.db.financial_reports.update_one(
            {"_id": ObjectId(report_id)},
            {"$set": updated_report}
        )
        if result.modified_count == 1:
            return {"message": "Financial report updated successfully"}
        return {"message": "Report not found or update failed"}, 404

    def delete(self, report_id):
        mongo.db.financial_reports.delete_one({"_id": ObjectId(report_id)})
        return {"message": "Report deleted successfully"}

class ImportFinancialReports(Resource):
    def allowed_file(self, filename):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in {"csv", "xlsx"}

    def post(self):
        file = request.files["file"]
        if file and self.allowed_file(file.filename):
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.filename.endswith(".xlsx"):
                df = pd.read_excel(file)

            records = df.to_dict("records")
            mongo.db.financial_reports.insert_many(records)
            return {"message": "Data imported successfully"}
        return {"message": "Invalid file format or upload failed"}, 400

class ClusterFinancialReports(Resource):
    def get(self):
        data = []
        for report in mongo.db.financial_reports.find():
            data_point = [
                report['roa'], report['roe'], report['eps'], 
                report['npm'], report['bv'], report['price_to_bv'], 
                report['pe_ratio'], report['de_ratio']
            ]
            data.append(data_point)

        data = np.array(data)

        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data)

        k = int(request.args.get('k', 3))  # Number of clusters, default to 3

        manhattan_distances = cdist(normalized_data, normalized_data, metric='cityblock')

        kmedoids = KMedoids(n_clusters=k, metric='precomputed')
        kmedoids.fit(manhattan_distances)

        cluster_assignments = kmedoids.labels_
        medoids_indices = kmedoids.medoid_indices_

        clusters = {}
        for cluster_id in range(k):
            cluster_data_indices = [i for i, label in enumerate(cluster_assignments) if label == cluster_id]
            clusters[f'Cluster {cluster_id}'] = {
                'Data Indices': cluster_data_indices.tolist(),
                'Medoid Index': int(medoids_indices[cluster_id])
            }

        return jsonify(clusters)

class FinancialReportDataRanges(Resource):
    def get(self):
        reports = list(mongo.db.financial_reports.find())

        data_ranges = {
            'highest': {},
            'medium': {},
            'lowest': {}
        }

        variables = ['roa', 'roe', 'eps', 'npm', 'bv', 'price_to_bv', 'pe_ratio', 'de_ratio']

        for variable in variables:
            sorted_reports = sorted(reports, key=lambda x: x[variable])
            data_ranges['highest'][variable] = sorted_reports[-5:]
            data_ranges['medium'][variable] = sorted_reports[len(sorted_reports) // 3: len(sorted_reports) // 3 + 5]
            data_ranges['lowest'][variable] = sorted_reports[:5]

        return jsonify(data_ranges)

api.add_resource(UserList, "/users")
api.add_resource(UserLogin, "/login")
api.add_resource(FinancialReport, "/financial-report")
api.add_resource(FinancialReportList, "/financial-reports")
api.add_resource(FinancialReportDetail, "/financial-report/<string:report_id>")
api.add_resource(ImportFinancialReports, "/import-financial-reports")
api.add_resource(ClusterFinancialReports, "/cluster-financial-reports")
api.add_resource(FinancialReportDataRanges, "/financial-report-data-ranges")

if __name__ == '__main__':
    app.run(debug=True, host='192.168.1.5', port=80)
